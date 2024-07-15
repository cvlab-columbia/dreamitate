from typing import Optional

import torchdata.datapipes.iter
import webdataset as wds
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from sgm.util import append_dims
import torch.nn.functional as F
import glob
import numpy as np
from tqdm import tqdm
import cv2
# from sgm.data import common

from torch.utils.data import DataLoader, Dataset, default_collate, DistributedSampler
import pytorch_lightning as pl
from einops import rearrange

import lovely_numpy
import lovely_tensors
from lovely_numpy import lo
from rich import print
lovely_tensors.monkey_patch()
np.set_printoptions(precision=3, suppress=True)

from PIL import Image
import os
import torch
from copy import copy
import open_clip
from torchvision.transforms import ToTensor
import random
import math

def resize_video(video_array, target_height, target_width):
    # Calculate aspect ratios
    original_height, original_width = video_array.shape[1:3]
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height

    # Crop to correct aspect ratio
    if original_aspect > target_aspect:
        # Crop width
        new_width = int(original_height * target_aspect)
        start = (original_width - new_width) // 2
        cropped_video = video_array[:, :, start: start + new_width, :]
    else:
        # Crop height
        new_height = int(original_width / target_aspect)
        start = (original_height - new_height) // 2
        cropped_video = video_array[:, start: start + new_height, :, :]

    # Resize video
    resized_video = np.zeros(
        (video_array.shape[0], target_height, target_width, video_array.shape[3])
    )
    for i, frame in enumerate(cropped_video):
        resized_video[i] = cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )

    return resized_video


class InMemoryVideoDataset(Dataset):
    def __init__(
        self,
        n_frames: int = 26,  # Typically model + 1.
        frame_width: int = 768,
        frame_height: int = 448,
        cond_aug: float = 0.02,
        data_list: str = "",
        data_dir: str = "",
        sample_fps: float = 6,
        vit_resolution=(224, 224),
    ):
        super().__init__()

        self.n_frames = n_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cond_aug = cond_aug
            
        image_list = []
        lines = open(data_list, 'r').readlines()
        lines = [[data_dir, item] for item in lines]    # data_dir is video_dir, item is lines in the txt file
        image_list.extend(lines)
        self.image_list = image_list
        self.n_videos = len(self.image_list)
        self.vit_resolution = vit_resolution
        self.sample_fps = sample_fps

    def __getitem__(self, i):
        try:
            data_dir, data_line = self.image_list[i]
            first_frame_top, first_frame_side, video_data, caption, motion_score = self._get_video_data(data_dir, data_line)
        except Exception as e:
            i = 0
            data_dir, data_line = self.image_list[i]
            first_frame_top, first_frame_side, video_data, caption, motion_score = self._get_video_data(data_dir, data_line)

        cond_aug = np.ones(shape=(self.n_frames-1,)) * self.cond_aug
        cond_frames_top = np.expand_dims(first_frame_top, axis=0).repeat(self.n_frames-1-12, axis=0) 
        cond_frames_side = np.expand_dims(first_frame_side, axis=0).repeat(self.n_frames-1-13, axis=0)

        cond_frames = np.concatenate([cond_frames_top, cond_frames_side], axis=0)   # cond_frames:(25, 3, frame_height, frame_width)

        # print(cond_frames_without_noise_top.shape, cond_frames_without_noise_side.shape, cond_frames_without_noise.shape)

        cond_frames = (
            cond_frames
            + self.cond_aug * np.random.randn(*cond_frames.shape))

        caption = open_clip.tokenize(caption)   # caption shape is (1, 77)
        cond_frames_without_noise = caption.numpy()
        cond_frames_without_noise = cond_frames_without_noise.repeat(self.n_frames-1, axis=0)

        # TODO: Hardcoded values for now -- should probably be customized.
        motion_bucket_id = np.ones(shape=(self.n_frames-1,), dtype=np.int32) * motion_score
        fps_id = np.ones(shape=(self.n_frames-1,), dtype=np.int32) * self.sample_fps
        image_only_indicator = np.zeros(shape=(1, self.n_frames-1,))

        return {
            "jpg": video_data.astype(np.float32),
            "cond_frames": cond_frames.astype(np.float32),
            "cond_frames_without_noise": cond_frames_without_noise,
            "cond_aug": cond_aug.astype(np.float32),
            "motion_bucket_id": motion_bucket_id,
            "fps_id": fps_id,
            "image_only_indicator": image_only_indicator.astype(np.float32),
        }
    
    def _get_video_data(self, data_dir, data_line):

        video_key, caption = data_line.split('|||')
        motion_score = float(200)
        video_path_top = os.path.join(data_dir, video_key)

        first_frame_top, video_data_top = self.get_video_data(video_path_top)
        
        video_path_side = os.path.join(data_dir, video_key.replace('top', 'side'))

        first_frame_side, video_data_side = self.get_video_data(video_path_side)

        video_data = np.concatenate((np.expand_dims(first_frame_top, axis=0), video_data_top, video_data_side), axis=0)

        # print(first_frame_top.shape, first_frame_side.shape, video_data_top.shape, video_data_side.shape, video_data.shape)

        return first_frame_top, first_frame_side, video_data, caption, motion_score

    def get_video_data(self, video_path):

        retry_count = 0
        while retry_count < 5:
            try:
                capture = cv2.VideoCapture(video_path)
                _fps = capture.get(cv2.CAP_PROP_FPS)
                _total_frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                stride = round(_fps / self.sample_fps)  # determine one frame in dataset is corresponding to how many frames in the video
                cover_frame_num = (stride * (self.n_frames-13))    # Determines the total number of frames in the original video to cover
                if _total_frame_num < cover_frame_num:
                    start_frame = 0
                    end_frame = _total_frame_num
                else:
                    start_frame = 0
                    end_frame = min(start_frame + cover_frame_num, _total_frame_num)

                pointer, frame_list, first_frame = 0, [], None
                while(True):
                    ret, frame = capture.read()
                    if (not ret) or (frame is None): break
                    if pointer < start_frame: 
                        pointer += 1
                        continue
                    if pointer >= end_frame: break
                    if (pointer - start_frame) % stride == 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        original_height, original_width = frame.shape[:2]
                        new_width = int(original_height * (self.frame_width/self.frame_height))
                        crop_start = (original_width - new_width) // 2
                        cropped_image = frame[:, crop_start:crop_start + new_width]
                        frame = cv2.resize(cropped_image, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)

                        frame = Image.fromarray(frame)
                        frame = ToTensor()(frame).numpy()
                        frame = frame * 2.0 - 1.0

                        if first_frame is None:
                            first_frame = frame
                        elif len(frame_list) < self.n_frames-1-13:
                            frame_list.append(frame)

                    pointer +=1 

                if first_frame is not None and len(frame_list) != 0:
                    break  # Break if the first frame is successfully obtained

                retry_count += 1

            except Exception as e:
                print(e)
                continue
        
        video_data = torch.zeros(self.n_frames-1-13, 3,  self.frame_height, self.frame_width).numpy()
        if len(frame_list)>0:
            video_data[:len(frame_list), ...] = frame_list

            # fill missing frames
            if len(frame_list) < self.n_frames-1-13:
                last_frame = frame_list[-1]  # Get the last frame from the frame list
                remaining_frames = self.n_frames-1-13 - len(frame_list)
                expanded_frame = np.expand_dims(last_frame, axis=0)
                repeated_frames = np.repeat(expanded_frame, remaining_frames, axis=0)
                video_data[len(frame_list):, ...] = repeated_frames

        return first_frame.astype(float), video_data

    def __len__(self):
        return self.n_videos


def collate_fn(example_list):
    collated = default_collate(example_list)
    batch = {k: rearrange(v, "b t ... -> (b t) ...") for (k, v) in collated.items()}
    batch["num_video_frames"] = 25
    return batch


class InMemoryVideoDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size=1, num_workers=1, shuffle=True,
            **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.dataset = InMemoryVideoDataset(**kwargs)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            collate_fn=collate_fn,
        )
