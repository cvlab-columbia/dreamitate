# Dreamitate: Real-World Visuomotor Policy Learning via Video Generation
### Arxiv 2024
### [Project Page](https://dreamitate.cs.columbia.edu/) | [Paper](https://dreamitate.cs.columbia.edu/assets/dreamitate_arxiv_v2.pdf) | [ArXiv](https://arxiv.org/abs/2406.16862)

[Junbang Liang](https://junbangliang.github.io/)<sup>*1</sup>, [Ruoshi Liu](https://ruoshiliu.github.io/)<sup>*1</sup>, [Ege Ozguroglu](https://www.cs.columbia.edu/~eo2464/)<sup>1</sup>, [Sruthi Sudhakar](https://sruthisudhakar.github.io/)<sup>1</sup>, [Achal Dave](https://www.achaldave.com/)<sup>2</sup>, [Pavel Tokmakov](https://pvtokmakov.github.io/home/)<sup>2</sup>, [Shuran Song](https://shurans.github.io/)<sup>3</sup>, [Carl Vondrick](https://www.cs.columbia.edu/~vondrick/)<sup>1</sup>

<sup>1</sup>Columbia University, <sup>2</sup>Toyota Research Institute, <sup>3</sup>Stanford University  
*Equal Contribution

<p align="center">
  <img width="100%" src="assets/animation_v4.gif">
</p>

##  Usage
###  Gradio Demo Inference
```
conda create -n dreamitate python=3.10
conda activate dreamitate
cd dreamitate
pip install -r requirements.txt
cd video_model
pip install .
pip install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata
```

Download image-conditioned stable video diffusion checkpoint released by [Stability AI](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and move `checkpoints` under the `video_model` folder:
```
wget https://dreamitate.cs.columbia.edu/assets/models/checkpoints.zip
```

Download the finetuned rotation task checkpoint and move `finetuned_models` under the `video_model` folder:

```
wget https://dreamitate.cs.columbia.edu/assets/models/finetuned_models.zip
```

Run our Gradio demo to generate videos of object rotation by using experiment photos from the `video_model/rotation_examples` directory as model inputs:
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/sampling/simple_video_sample_gradio.py
```
Alternatively, you can use online images of object against a black background as model inputs, which is less suitable but can work for this demonstration.
Note that this app uses around 70 GB of VRAM, so it may not be possible to run it on any GPU.

### Training Script

Download image-conditioned stable video diffusion checkpoint released by [Stability AI](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and move `checkpoints` under the `video_model` folder:
```
wget https://dreamitate.cs.columbia.edu/assets/models/checkpoints.zip
```

Download the rotation task dataset and move `dataset` under the `video_model` folder:
```
wget https://dreamitate.cs.columbia.edu/assets/models/dataset.zip
```

Run training command:  
```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base=configs/basile_svd_finetune.yaml --name=ft1 --seed=24 --num_nodes=1 --wandb=0 lightning.trainer.devices="0,1,2,3"
```

Note that this training script is set for an 4-GPU system, each with 80GB of VRAM. Empirically a batch size of 4 is found to produce good results for training our model, but training with a batch size of 1 can work as well.

### Tool Tracking

Download the pretrained models and move `megapose-models` under the `megapose/examples` folder:
```
wget https://dreamitate.cs.columbia.edu/assets/models/megapose-models.zip
```

Set environment variables:
```
cd dreamitate/megapose
export MEGAPOSE_DIR=$(pwd) && export MEGAPOSE_DATA_DIR=$(pwd)/examples && export megapose_directory_path=$(pwd)/src && export PYTHONPATH="$PYTHONPATH:$megapose_directory_path"
```

Run tracking on left end-effector:  
```
CUDA_VISIBLE_DEVICES=0 python -m megapose.scripts.run_video_tracking_on_rotation_example_stereo_left --data_dir "experiments/rotation/demo_005"
```

Run tracking on right end-effector:  
```
CUDA_VISIBLE_DEVICES=0 python -m megapose.scripts.run_video_tracking_on_rotation_example_stereo_right --data_dir "experiments/rotation/demo_005"
```

##  Acknowledgement
This repository is based on [Stable Video Diffusion](https://github.com/Stability-AI/generative-models), [Generative Camera Dolly](https://gcd.cs.columbia.edu/), and [MegaPose](https://github.com/megapose6d/megapose6d). We would like to thank the authors of these work for publicly releasing their code. We would like to thank Basile Van Hoorick and Kyle Sargent of [Generative Camera Dolly](https://gcd.cs.columbia.edu/) for providing the video model training code and their helpful feedback.

We would like to thank Paarth Shah and Dian Chen for many helpful discussions. This research is based on work partially supported by the Toyota Research Institute and the NSF NRI Award #2132519.


##  Citation
```
@misc{liang2024dreamitate,
      title={Dreamitate: Real-World Visuomotor Policy Learning via Video Generation}, 
      author={Junbang Liang and Ruoshi Liu and Ege Ozguroglu and Sruthi Sudhakar and Achal Dave and Pavel Tokmakov and Shuran Song and Carl Vondrick},
      year={2024},
      eprint={2406.16862},
      archivePrefix={arXiv},
      primaryClass={id='cs.RO' full_name='Robotics' is_active=True alt_name=None in_archive='cs' is_general=False description='Roughly includes material in ACM Subject Class I.2.9.'}
}
```
