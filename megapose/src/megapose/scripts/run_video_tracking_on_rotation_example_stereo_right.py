# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image

# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

import trimesh
import cv2
import torch
from collections import deque
from scipy.spatial.transform import Rotation

import time
import copy

logger = get_logger(__name__)


# Initialize bbox as a global variable
bbox_top = np.array([270, 190, 370, 290])
bbox_side = np.array([270, 190, 370, 290])

def get_camera_tf():   # transforms top camera to side camera

    A_T_B = np.array([[1.0, 0.0, 0.0, 0.0],     # 22.5 degrees rotation 
                     [0.0, 0.9238795, 0.3826834, 0.0],
                     [0.0, -0.3826834, 0.9238795, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])
    
    translation = 0.6596569

    B_T_C = np.array([[1.0, 0.0, 0.0, 0.0],     # 0.6596569 m translation
                     [0.0, 1.0, 0.0, -translation],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])
    
    A_T_D = A_T_B @ B_T_C @ A_T_B

    return A_T_D

def compute_projection_line(P_3D):
    """
    Compute the projection line (ray) from the camera to the 3D point.
    
    Parameters:
    P_3D (tuple): A 3D point in space (X, Y, Z) relative to the camera.
    
    Returns:
    tuple: The origin (O) and direction vector (D) of the projection line.
    """
    # Camera origin, assuming the camera is at the origin of the coordinate system
    O = [0, 0, 0]
    
    # Direction vector from the camera to the point is just the point's coordinates
    # assuming the camera is looking in the direction of the Z-axis and located at the origin
    D = P_3D
    
    return np.array(O), np.array(D)

def transform_point(T, P_3D_prime):
    """
    Transform a 3D point from the second camera's coordinate system to the reference camera's coordinate system.
    
    Parameters:
    T (numpy.ndarray): 4x4 homogeneous transformation matrix of the second camera relative to the reference camera.
    P_3D_prime (tuple): A 3D point (X', Y', Z') observed in the second camera's coordinate system.
    
    Returns:
    numpy.ndarray: The transformed 3D point in the reference camera's coordinate system.
    """
    P_4D_prime = np.array([*P_3D_prime, 1])  # Convert to homogeneous coordinates
    P_4D = np.dot(T, P_4D_prime)  # Apply transformation
    return P_4D[:3]  # Return as 3D point

def compute_line_equation(T, P_3D_prime):
    """
    Compute the line equation that passes through the 3D point and the second camera, as seen from the reference camera's coordinate system.
    
    Parameters:
    T (numpy.ndarray): 4x4 homogeneous transformation matrix of the second camera relative to the reference camera.
    P_3D_prime (tuple): A 3D point (X', Y', Z') observed in the second camera's coordinate system.
    
    Returns:
    tuple: The origin (O) and direction vector (D) of the line.
    """
    # Transform the 3D point to the reference camera's coordinate system
    P_3D = transform_point(T, P_3D_prime)
    
    # Extract the camera position from the transformation matrix
    O = T[:3, 3]
    
    # The direction vector from the camera to the point
    D = P_3D - O
    
    return O, D


def find_closest_points(P1, D1, P2, D2):
    """
    Finds the closest points on two lines defined by points P1, P2 and direction vectors D1, D2.

    Parameters:
    P1, P2 (numpy.ndarray): Points on the first and second lines.
    D1, D2 (numpy.ndarray): Direction vectors of the first and second lines.

    Returns:
    numpy.ndarray: The closest points on each of the two lines.
    """
    # Compute some dot products we'll need
    D1_dot_D1 = np.dot(D1, D1)
    D2_dot_D2 = np.dot(D2, D2)
    D1_dot_D2 = np.dot(D1, D2)
    P_diff = P1 - P2
    D1_dot_P_diff = np.dot(D1, P_diff)
    D2_dot_P_diff = np.dot(D2, P_diff)

    # Compute the denominators for the equations to solve t and s
    denom = D1_dot_D1 * D2_dot_D2 - D1_dot_D2 ** 2

    # Ensure lines are not parallel to avoid division by zero
    if denom == 0:
        raise ValueError("Lines are parallel")

    # Solve for the parameters t and s that minimize the distance between the two lines
    t = (D1_dot_D2 * D2_dot_P_diff - D2_dot_D2 * D1_dot_P_diff) / denom
    s = (D1_dot_D1 * D2_dot_P_diff - D1_dot_D2 * D1_dot_P_diff) / denom

    # Compute the closest points on each line
    Q1 = P1 + t * D1
    Q2 = P2 + s * D2

    return Q1, Q2

def compute_midpoint_between_lines(P1, D1, P2, D2):
    """
    Computes the midpoint between the closest points of two lines.

    Parameters:
    P1, P2 (numpy.ndarray): Points on the first and second lines.
    D1, D2 (numpy.ndarray): Direction vectors of the first and second lines.

    Returns:
    numpy.ndarray: The midpoint between the closest points on the two lines.
    """
    Q1, Q2 = find_closest_points(P1, D1, P2, D2)
    midpoint = (Q1 + Q2) / 2
    return midpoint


def merge_stereo_views(top_pose, side_pose, top_T_side):
    O_top, D_top = compute_projection_line(top_pose[:3, 3])
    O_side, D_side = compute_line_equation(top_T_side, side_pose[:3, 3])
    midpoint = compute_midpoint_between_lines(O_top, D_top, O_side, D_side)

    avg_rotation_tf = average_transforms([top_pose, top_T_side@side_pose])

    avg_tf = np.eye(4)

    avg_tf[:3,:3] = avg_rotation_tf[:3,:3]
    avg_tf[:3, 3] = midpoint

    return avg_tf

def inverse_homogeneous_matrix(matrix):

    R = matrix[:3, :3]  # Extract the 3x3 rotation matrix
    t = matrix[:3, 3]   # Extract the translation vector
    
    R_inv = R.T  # Compute the inverse (transpose) of the rotation matrix
    t_inv = -np.dot(R_inv, t)  # Compute the inverse translation vector
    
    # Construct the inverse matrix
    inverse_matrix = np.zeros_like(matrix)
    inverse_matrix[:3, :3] = R_inv
    inverse_matrix[:3, 3] = t_inv
    inverse_matrix[3, 3] = 1  # Set the bottom-right element to 1
    
    return inverse_matrix

def load_observation(
    example_dir: Path,
    load_depth: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())

    rgb = np.array(Image.open(example_dir / "image_rgb.png"), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = np.array(Image.open(example_dir / "image_depth.png"), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data

def load_video_observation(
    example_dir: Path,
    video_dir: str,
    load_depth: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())

    video_frames = read_video_frames(video_dir)
    assert video_frames[0].shape[:2] == camera_data.resolution

    # rgb = np.array(Image.open(example_dir / "image_rgb.png"), dtype=np.uint8)
    # assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = np.array(Image.open(example_dir / "image_depth.png"), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return video_frames, depth, camera_data

def read_video_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    if not cap.isOpened():
        print("Error opening video file")
        return frames

    skip_first_frames = 0
    count = 0
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret:
            original_height, original_width = frame.shape[:2]
            new_width = int(original_height * (768/448))
            crop_start = (original_width - new_width) // 2
            cropped_image = frame[:, crop_start:crop_start + new_width]
            frame = cv2.resize(cropped_image, (768, 448))

            # Convert the color space from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

            if count == 0 or count >= skip_first_frames:
                frames.append(frame_rgb)
            
            count += 1
        else:
            break
        
    # When everything done, release the video capture object
    cap.release()

    return frames

def load_observation_tensor(
    example_dir: Path,
    load_depth: bool = False,
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(example_dir, load_depth)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation


def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def load_detections(
    example_dir: Path,
) -> DetectionsType:
    input_object_data = load_object_data(example_dir / "inputs/object_data.json")
    detections = make_detections_from_object_data(input_object_data).cuda()
    return detections


def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset

def compute_change_mask(frame1, frame2, threshold=30):

    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
    # Compute absolute difference between the two frames
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply a threshold to get the binary mask
    _, binary_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Create a 3-channel mask from the binary mask
    binary_mask_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    
    # Apply the binary mask to the second frame to retain colors at changed pixels
    change_mask = cv2.bitwise_and(frame2, binary_mask_rgb)
    
    return change_mask

def create_hand_mask(im):

    im_ycrcb = cv2.cvtColor(im, cv2.COLOR_RGB2YCR_CB)

    # Adjusted skin color range to increase sensitivity
    skin_ycrcb_mint = np.array((0, 133, 77))
    skin_ycrcb_maxt = np.array((255, 173, 135))
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)

    contours, _ = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(im)

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        # Lowered the area threshold to detect smaller skin regions
        if area > 500:
            cv2.drawContours(mask, contours, i, (255, 255, 255), cv2.FILLED)
    
    return mask

def compute_hand_change_mask(frame1, frame2, threshold=30):

    # Compute the change mask
    change_mask = compute_change_mask(frame1, frame2, threshold)

    # cv2.imshow("img", change_mask)
    # cv2.waitKey()

    # Perform hand masking on the difference image
    hand_mask = create_hand_mask(change_mask)

    # Invert the hand mask to get non-hand regions
    non_hand_mask = cv2.bitwise_not(hand_mask)

    # Apply the non-hand mask to the change mask to get the final mask
    final_mask = cv2.bitwise_and(change_mask, non_hand_mask)

    # cv2.imshow("img", final_mask)
    # cv2.waitKey()

    # Perform erosion followed by dilation
    kernel = np.ones((3, 3), np.uint8)  # Define the kernel size
    final_mask = cv2.erode(final_mask, kernel, iterations=1)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)

    # cv2.imshow("img", final_mask)
    # cv2.waitKey()

    # Create a mask where any channel intensity is greater than 120
    intensity_mask = (final_mask[:,:,0] > 130) | (final_mask[:,:,1] > 130) | (final_mask[:,:,2] > 130)

    # Convert the mask to an unsigned 8-bit integer array
    intensity_mask = intensity_mask.astype(np.uint8)

    # Multiply the mask with the original image
    final_mask = final_mask * intensity_mask[:,:,None]

    return final_mask

def find_and_visualize_colored_regions(image):
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Threshold the image to create a binary image
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Apply morphological closing operation to fill holes in the colored regions
    kernel = np.ones((40, 40), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours in the binary image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store bounding boxes
    bounding_boxes = []
    
    # Loop over the contours
    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append([x, y, x + w, y + h])
    
    # Sort the bounding boxes by area (largest first)
    bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    
    # If less than 2 bounding boxes, add zero-valued bounding boxes
    while len(bounding_boxes) < 2:
        bounding_boxes.append([0, 0, 0, 0])

    # Keep only the two largest bounding boxes
    bounding_boxes = bounding_boxes[:2]
    
    # Sort the bounding boxes by the x-coordinate (leftmost first)
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])

    # Create a list of the two bounding boxes
    bounding_boxes = [bounding_boxes[0], bounding_boxes[1]]
    
    # # Display the image with bounding boxes
    # cv2.imshow('Bounding Boxes', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return np.array(bounding_boxes[1])

def estiamte_new_pose(bbox, K, original_pose):

    u = int((bbox[0] + bbox[2]) / 2)
    v = int((bbox[1] + bbox[3]) / 2)

    # Extract the translation vector from the original pose matrix
    original_translation = original_pose[:3, 3]
    
    # Calculate depth as the Euclidean distance from the camera to the object
    Z = np.linalg.norm(original_translation)
    
    # Create the homogeneous coordinate of the pixel
    uv1 = np.array([u, v, 1])
    
    # Invert the camera matrix
    K_inv = np.linalg.inv(K)
    
    # Back-project the point to 3D space
    x, y, z = Z * (K_inv @ uv1)
    
    # Calculate new translation
    new_translation = np.array([x, y, Z])
    
    # New pose: same rotation, updated translation
    new_pose = original_pose.copy()
    new_pose[:3, 3] = new_translation
    
    return new_pose

def find_mp4_files(folder_path):
    # Walk through all directories and files in the specified folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mp4"):
                # Construct the full path to the file
                full_path = os.path.join(root, file)
                return full_path
    
    print("No video found in folder")
    exit()


def run_inference(
    example_dir: Path,
    model_name: str,
    data_dir: str,
) -> None:
    
    mesh_path = find_mesh_path(example_dir)

    point_cloud = obj_to_point_cloud(str(mesh_path), num_points=4096) / 1000

    object_dataset = make_object_dataset(example_dir)

    detections_top = load_detections(example_dir).cuda()
    detections_side = load_detections(example_dir).cuda()

    model_info = NAMED_MODELS[model_name]

    pose_estimator_top = load_named_model(model_name, object_dataset).cuda()
    pose_estimator_side = load_named_model(model_name, object_dataset).cuda()

    video_path = find_mp4_files(data_dir)

    video_frames_top, depth_top, camera_data_top = load_video_observation(example_dir, video_path, load_depth=model_info["requires_depth"])
    
    camera_data_side = camera_data_top
    video_frames_side = video_frames_top[13:]
    video_frames_top = video_frames_top[0:13]

    camera_data_top.K = np.array([[566.37337239,   0.        , 390.48050673],
                                    [  0.        , 565.92041558, 232.6177436 ],
                                    [  0.        ,   0.        ,   1.        ]])
    
    camera_data_side.K = np.array([[569.55341254,   0.        , 379.1203559 ],
                                [  0.        , 569.43598633, 232.61307237],
                                [  0.        ,   0.        ,   1.        ]])

    # Filter files that contain the specified substring
    image_files = [file for file in os.listdir(data_dir) if "side" in file.lower()]
    
    frame = cv2.imread(os.path.join(data_dir, image_files[0]))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_height, original_width = frame.shape[:2]
    new_width = int(original_height * (768/448))
    crop_start = (original_width - new_width) // 2
    cropped_image = frame[:, crop_start:crop_start + new_width]
    frame = cv2.resize(cropped_image, (768, 448))
    video_frames_side.insert(0, frame)

    global bbox_top
    global bbox_side

    skip_frames = 3

    mask_rgb = compute_hand_change_mask(video_frames_top[0], video_frames_top[skip_frames+1], 40)
    bbox_top = find_and_visualize_colored_regions(mask_rgb)
    mask_rgb = compute_hand_change_mask(video_frames_side[0], video_frames_side[skip_frames+1], 40)
    bbox_side = find_and_visualize_colored_regions(mask_rgb)

    video_frames_top_first = video_frames_top[0]
    video_frames_side_first = video_frames_side[0]

    video_frames_top.pop(0)
    video_frames_side.pop(0)

    video_frames_top = video_frames_top[skip_frames:]
    video_frames_side = video_frames_side[skip_frames:]

    cv2.namedWindow("Click to Draw Detection Box")
    cv2.setMouseCallback("Click to Draw Detection Box", draw_rectangle_top)

    print("Click to draw detection box and press 'q' to continue")
    while cv2.waitKey(1) != ord("q"):
        # Create a fresh copy of the original image for each iteration
        annotated_image_top = video_frames_top[0].copy()
        cv2.rectangle(
            annotated_image_top,
            tuple(bbox_top[0:2]),
            tuple(bbox_top[2:4]),
            (0, 255, 0),
            2,
        )

        cv2.imshow("Click to Draw Detection Box", annotated_image_top)
    
    cv2.setMouseCallback("Click to Draw Detection Box", draw_rectangle_side)

    print("Click to draw detection box and press 'q' to continue")
    while cv2.waitKey(1) != ord("q"):
        # Create a fresh copy of the original image for each iteration
        annotated_image_side = video_frames_side[0].copy()
        cv2.rectangle(
            annotated_image_side,
            tuple(bbox_side[0:2]),
            tuple(bbox_side[2:4]),
            (0, 255, 0),
            2,
        )

        cv2.imshow("Click to Draw Detection Box", annotated_image_side)

    cv2.destroyAllWindows()

    i = 0
    iterations_per_frame = 2

    pose_list_top = deque(maxlen=2)
    pose_score_top = deque(maxlen=2)
    output_pose_list_top = []
    pose_list_history_top = []
    pose_list_side = deque(maxlen=2)
    pose_score_side = deque(maxlen=2)
    output_pose_list_side = []
    pose_list_history_side = []
    min_score_top = 1.0
    min_score_frame_top = annotated_image_top
    min_score_side = 1.0
    min_score_frame_side = annotated_image_top

    while i < iterations_per_frame*len(video_frames_top):
        rgb_image_intance_top = video_frames_top[i//iterations_per_frame]
        rgb_image_intance_side = video_frames_side[i//iterations_per_frame]
        
        detections_top.bboxes = torch.tensor(
            np.array([bbox_top]), dtype=torch.int32, device="cuda"
        )
        detections_side.bboxes = torch.tensor(
            np.array([bbox_side]), dtype=torch.int32, device="cuda"
        )

        observation_top = ObservationTensor.from_numpy(
            rgb_image_intance_top, depth=None, K=camera_data_top.K
        ).cuda()

        observation_side = ObservationTensor.from_numpy(
            rgb_image_intance_side, depth=None, K=camera_data_side.K
        ).cuda()

        if i == 0:
            # Run the pose estimation model
            output_top, _ = pose_estimator_top.run_inference_pipeline(
                observation_top, detections=detections_top, **model_info["inference_parameters"]
            )
            output_side, _ = pose_estimator_side.run_inference_pipeline(
                observation_side, detections=detections_side, **model_info["inference_parameters"]
            )

        else:
            # Run the pose estimation model with stored_output
            output_top, _ = pose_estimator_top.run_inference_pipeline(
                observation_top,
                detections=detections_top,
                **model_info["inference_parameters"],
                coarse_estimates=output_top,
            )
            # Run the pose estimation model with stored_output
            output_side, _ = pose_estimator_side.run_inference_pipeline(
                observation_side,
                detections=detections_side,
                **model_info["inference_parameters"],
                coarse_estimates=output_side,
            )

        threshold = 0.8

        if i != 0:

            if output_top.infos['pose_score'][0] < threshold:
                mask_rgb = compute_hand_change_mask(video_frames_top_first, rgb_image_intance_top, 40)
                bbox_top = find_and_visualize_colored_regions(mask_rgb)
                cv2.namedWindow("Click to Draw Detection Box")
                cv2.setMouseCallback("Click to Draw Detection Box", draw_rectangle_top)

                print("Click to draw detection box and press 'q' to continue")
                while cv2.waitKey(1) != ord("q"):
                    # Create a fresh copy of the original image for each iteration
                    annotated_image_top = rgb_image_intance_top.copy()
                    cv2.rectangle(
                        annotated_image_top,
                        tuple(bbox_top[0:2]),
                        tuple(bbox_top[2:4]),
                        (0, 255, 0),
                        2,
                    )

                    cv2.imshow("Click to Draw Detection Box", annotated_image_top)

                cv2.destroyAllWindows()

                new_estimated_pose = estiamte_new_pose(bbox=bbox_top, K=camera_data_top.K, original_pose=pose_list_history_top[-1])
            
                output_top.poses = torch.from_numpy(new_estimated_pose).unsqueeze(0).float().cuda()
                output_top, _ = pose_estimator_top.run_inference_pipeline(
                    observation_top,
                    detections=detections_top,
                    **model_info["inference_parameters"],
                    coarse_estimates=output_top,
                )  
                print("top corrected")

            if output_side.infos['pose_score'][0] < threshold:
                mask_rgb = compute_hand_change_mask(video_frames_side_first, rgb_image_intance_side, 40)
                bbox_side = find_and_visualize_colored_regions(mask_rgb)

                cv2.namedWindow("Click to Draw Detection Box")
                cv2.setMouseCallback("Click to Draw Detection Box", draw_rectangle_side)

                print("Click to draw detection box and press 'q' to continue")
                while cv2.waitKey(1) != ord("q"):
                    # Create a fresh copy of the original image for each iteration
                    annotated_image_side = rgb_image_intance_side.copy()
                    cv2.rectangle(
                        annotated_image_side,
                        tuple(bbox_side[0:2]),
                        tuple(bbox_side[2:4]),
                        (0, 255, 0),
                        2,
                    )

                    cv2.imshow("Click to Draw Detection Box", annotated_image_side)

                cv2.destroyAllWindows()

                new_estimated_pose = estiamte_new_pose(bbox=bbox_side, K=camera_data_side.K, original_pose=pose_list_history_side[-1])

                output_side.poses = torch.from_numpy(new_estimated_pose).unsqueeze(0).float().cuda()
                output_side, _ = pose_estimator_side.run_inference_pipeline(
                    observation_side,
                    detections=detections_side,
                    **model_info["inference_parameters"],
                    coarse_estimates=output_side,
                )
                print("side corrected")

        pose_top = np.array(output_top.poses.cpu().numpy()[0])
        pose_side = np.array(output_side.poses.cpu().numpy()[0])

        pose_score_top.append(output_top.infos['pose_score'][0])
        pose_score_side.append(output_side.infos['pose_score'][0])

        pose_score_top_avg = sum(pose_score_top) / len(pose_score_top)
        pose_score_side_avg = sum(pose_score_side) / len(pose_score_side)

        print(f"pose_score_top_avg: {pose_score_top_avg}")
        print(f"pose_score_side_avg: {pose_score_side_avg}")

        pose_list_top.append(pose_top)
        pose_top = average_transforms(pose_list_top)
        output_top.poses = torch.from_numpy(pose_top).unsqueeze(0).float().cuda()

        pose_list_side.append(pose_side)
        pose_side = average_transforms(pose_list_side)
        output_side.poses = torch.from_numpy(pose_side).unsqueeze(0).float().cuda()

        masked_image_top, new_bbox = mask_object_in_image(
            point_cloud, pose_top[:3], rgb_image_intance_top, camera_data_top.K
        )

        masked_image_side, new_bbox = mask_object_in_image(
            point_cloud, pose_side[:3], rgb_image_intance_side, camera_data_side.K
        )

        if (i%iterations_per_frame == iterations_per_frame-1):
            output_pose_list_top.append(np.around(pose_top, decimals=5).tolist())
            output_pose_list_side.append(np.around(pose_side, decimals=5).tolist())
            pose_list_history_top.append(pose_top)
            pose_list_history_side.append(pose_side)
            
            pose_list_top.clear()
            pose_list_side.clear()
            pose_score_top.clear()
            pose_score_side.clear()

            if min_score_top > pose_score_top_avg:
                min_score_top = pose_score_top_avg
                min_score_frame_top = masked_image_top
            
            if min_score_side > pose_score_side_avg:
                min_score_side = pose_score_side_avg
                min_score_frame_side = masked_image_side

            cv2.imshow(f"{video_path}", cv2.hconcat([masked_image_top, masked_image_side]))
            cv2.waitKey(1)

        i = i + 1

    ret = True

    print("Press any key to save tracking trajectory relative to top camera (view 1)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ret, output_pose_list_top, output_pose_list_side


def find_mesh_path(
    example_dir: Path,
) -> Path:
    object_dirs = (example_dir / "meshes").iterdir()
    # Iterate through individual object paths as PosixPath objects in the objects directory
    for object_dir in object_dirs:
        # Retrive object name
        label = object_dir.name
        mesh_path = None
        # Get the path to the object mesh model
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"

    return mesh_path

def obj_to_point_cloud(obj_file, num_points=4096):
    scene_or_mesh = trimesh.load_mesh(obj_file, force="mesh")
    
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh

    points, _ = trimesh.sample.sample_surface_even(mesh, num_points)
    return points

def draw_rectangle_top(event, x, y, flags, param):
    global bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox_top[0] = int(x)
        bbox_top[1] = int(y)
        print("cv2.EVENT_LBUTTONDOWN")

    elif event == cv2.EVENT_LBUTTONUP:
        bbox_top[2] = int(x)
        bbox_top[3] = int(y)
        print("cv2.EVENT_LBUTTONUP")

    return None

def draw_rectangle_side(event, x, y, flags, param):
    global bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox_side[0] = int(x)
        bbox_side[1] = int(y)
        print("cv2.EVENT_LBUTTONDOWN")

    elif event == cv2.EVENT_LBUTTONUP:
        bbox_side[2] = int(x)
        bbox_side[3] = int(y)
        print("cv2.EVENT_LBUTTONUP")

    return None

def mask_object_in_image(point_cloud, pose_3d, rgb_image, K):
    # Project the 3D points onto the 2D image
    projected_points = project_points(point_cloud, pose_3d, K)
    # Create an empty mask with the same dimensions as the input image
    mask = np.zeros_like(rgb_image)

    # Round and cast the projected points to integer indices
    int_points = np.round(projected_points).astype(int)
    # Filter out points that are outside the image boundaries
    valid_indices = np.logical_and(
        np.logical_and(0 <= int_points[:, 0], int_points[:, 0] < rgb_image.shape[1]),
        np.logical_and(0 <= int_points[:, 1], int_points[:, 1] < rgb_image.shape[0]),
    )
    # Update the mask with white (255) at the valid projected points
    mask[int_points[valid_indices, 1], int_points[valid_indices, 0]] = [255, 255, 255]

    bbox = np.array(
        [
            min(int_points[valid_indices, 0]) - 20,
            min(int_points[valid_indices, 1]) - 20,
            max(int_points[valid_indices, 0]) + 20,
            max(int_points[valid_indices, 1]) + 20,
        ]
    ).astype(int)

    kernel_size = 3

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    alpha = 0.5
    blended_image = cv2.addWeighted(rgb_image, 1 - alpha, mask, alpha, 0)
    
    # Draw XYZ axes
    axis_length = 0.1  # Length of the axes in meters (or any unit consistent with point_cloud)
    axes_points = np.array([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
    projected_axes_points = project_points(axes_points, pose_3d, K).astype(int)

    # Draw lines for the XYZ axes
    # X-axis in red, Y-axis in green, Z-axis in blue
    cv2.line(blended_image, tuple(projected_axes_points[0]), tuple(projected_axes_points[1]), (255, 0, 0), 2)
    cv2.line(blended_image, tuple(projected_axes_points[0]), tuple(projected_axes_points[2]), (0, 255, 0), 2)
    cv2.line(blended_image, tuple(projected_axes_points[0]), tuple(projected_axes_points[3]), (0, 0, 255), 2)

    return cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR), bbox

def project_points(point_cloud, pose_3d, mtx):
    # Convert point_cloud to homogeneous coordinates
    point_cloud_h = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # Perform matrix multiplications
    point_3d = pose_3d @ point_cloud_h.T
    point_2d = mtx @ point_3d

    # Convert back to Cartesian coordinates
    point_2d_cartesian = point_2d[:2] / point_2d[2]

    return point_2d_cartesian.T

def average_transforms(transforms):
    rotations = []
    translations = []

    # Extract rotations and translations
    for T in transforms:
        R_, t_ = T[:3, :3], T[:3, 3]
        rotations.append(Rotation.from_matrix(R_).as_quat())
        translations.append(t_)

    # Average the rotations
    mean_quat = Rotation.from_quat(rotations).mean().as_quat()
    mean_R = Rotation.from_quat(mean_quat).as_matrix()

    # Use SVD to ensure the mean rotation matrix is orthogonal
    U, _, Vt = np.linalg.svd(mean_R, full_matrices=True)
    mean_R_orthogonal = np.dot(U, Vt)

    # Average the translations
    mean_t = np.mean(translations, axis=0)

    # Construct the mean transformation matrix with the orthogonalized rotation
    mean_T = np.eye(4)
    mean_T[:3, :3] = mean_R_orthogonal
    mean_T[:3, 3] = mean_t

    return mean_T


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run video tracking on rotation example for the left stereo camera')
    parser.add_argument('--data_dir', type=str, default="experiments/rotation/demo_001",
                        help='Directory to process data from.')
    
    args = parser.parse_args()

    data_dir = args.data_dir

    set_logging_level("info")

    example_dir = LOCAL_DATA_DIR / "tracking_box_v3_768_448_right"

    ret = False
    while not ret:
        ret, pose_list_top, pose_list_side = run_inference(example_dir, "megapose-1.0-RGB-multi-hypothesis", data_dir)
        if not ret:
            print(f"Failed to process {data_dir}, retrying...")

    right_tf = []

    pose_list_top = np.array(pose_list_top)
    pose_list_side = np.array(pose_list_side)

    for i in range(len(pose_list_top)):
        output_pose = merge_stereo_views(pose_list_top[i], pose_list_side[i], get_camera_tf())
        right_tf.append(output_pose)
    
    np.savez(os.path.join(data_dir, "video_poses_right.npz"), right_tf=np.array(right_tf))



