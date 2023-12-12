import os
import multiprocessing as mp
import ffmpeg

import torch
import torchvision
import numpy as np
from tqdm import tqdm
from munch import Munch

from exp import ex
from ckpt import load_ckpt
from model import get_extractor
from optimizer import get_optimizer
from utils import prepare_batch, get_all

from metrics.log import Logger, Metric
from metrics.wild360 import calc_score, visualize_heatmap, get_gt_heatmap

import cv2


'''Configs: '''

# DATASET_NAME = 'Wu_MMSys_17'
# DATASET_NAME = 'David_MMSys_18'
# DATASET_NAME = 'Xu_CVPR_18'
DATASET_NAME = 'Nasrabadi_MMSys_19'

SALMAP_SHAPE = None  # saliency_src; 不进行resize, 直接保存最原始的模型输出 (224x448);
# SALMAP_SHAPE = (64, 128)  # saliency_paver;
# SALMAP_SHAPE = (128, 256)  # saliency_paver_mid;
# SALMAP_SHAPE = (224, 448)  # saliency_paver_big;

SALMAP_FOLDER = 'saliency_src' if SALMAP_SHAPE is None else f'saliency_{SALMAP_SHAPE[0]}x{SALMAP_SHAPE[1]}'


if DATASET_NAME == 'Wu_MMSys_17':
    VIDEO_DIR = '/home/gbq/datasets/vr-dataset/vid-prep/'
    VIDEO_NAMES = [video_name for video_name in os.listdir(VIDEO_DIR) if video_name.startswith('1-')]
    OUTPUT_DIR = f'../output/{DATASET_NAME}/{SALMAP_FOLDER}'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SAMPLING_RATE = 0.2
    denominator = int(SAMPLING_RATE/0.2)
    NUM_SAMPLES_PER_VIDEO = {  # 根据实际视频中能提取出的所有帧的最大播放时间点确定;
        '7' : 816 // denominator,
        '8' : 1436 // denominator,
        '1' : 1000 // denominator,
        '4' : 1028 // denominator,
        '6' : 2251 // denominator,
        '3' : 862 // denominator,
        '2' : 1466 // denominator,
        '0' : 821 // denominator,
        '5' : 3275 // denominator,
    }


elif DATASET_NAME == 'David_MMSys_18':
    VIDEO_DIR = '/dataset/saliency-datasets/Salient360/Videos/Stimuli/'
    OUTPUT_DIR = f'../output/{DATASET_NAME}/{SALMAP_FOLDER}/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SAMPLING_RATE = 0.2
    VIDEO_NAMES = [video_name for video_name in os.listdir(VIDEO_DIR) if video_name.endswith('.mp4')]  # 从VIDEO_DIR中的获取视频名称
    NUM_SAMPLES_PER_VIDEO = { video_name : 100 for video_name in VIDEO_NAMES }


elif DATASET_NAME == 'Xu_CVPR_18':
    VIDEO_DIR = '/home/gbq/datasets/Xu_CVPR_18/videos/'
    OUTPUT_DIR = f'../output/{DATASET_NAME}/{SALMAP_FOLDER}/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SAMPLING_RATE = 0.2
    VIDEO_NAMES = [video_name for video_name in os.listdir(VIDEO_DIR) if video_name.endswith('.mp4')]  # 从VIDEO_DIR中的获取视频名称
    def get_num_samples_per_video(sample_data_folder):
        num_samples_per_video = {}
        for video_name in VIDEO_NAMES:
            video_name = video_name.split('.')[0]
            # 每个视频文件夹中均包含多个用户的数据, 每个用户的数据文件是没有后缀但是可以作为csv读取的文件; 对于一个视频, 需要计算其每个用户数据的行数, 并将最大行数作为该视频的num_samples;
            num_samples = 0
            for user_data_file in os.listdir(sample_data_folder + video_name):
                with open(sample_data_folder + video_name + '/' + user_data_file, 'r') as f:
                    num_samples = max(num_samples, len(f.readlines()))
            num_samples_per_video[video_name] = num_samples
        return num_samples_per_video
    NUM_SAMPLES_PER_VIDEO = get_num_samples_per_video('/home/gbq/datasets/Xu_CVPR_18/sampled_dataset/')


elif DATASET_NAME == 'Nasrabadi_MMSys_19':
    VIDEO_DIR = '/dataset/uniformed-vp-datasets/Nasrabadi_MMSys_19/dataset/Videos/'
    OUTPUT_DIR = f'../output/{DATASET_NAME}/{SALMAP_FOLDER}/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SAMPLING_RATE = 0.2
    VIDEO_NAMES = [video_name for video_name in os.listdir(VIDEO_DIR) if video_name.endswith('.mp4')]  # 从VIDEO_DIR中的获取视频名称
    NUM_SAMPLES_PER_VIDEO = { video_name : 300 for video_name in VIDEO_NAMES }


else:
    raise NotImplementedError


@ex.capture()
def demo(log_path, config_dir, max_epoch, lr, clip_length, num_workers,
            display_config, num_frame, eval_start_epoch, save_model,
            input_video, model_config, input_format):
    # Extract feature from image
    model_config = Munch(model_config)

    feature_extractor = get_extractor()
    feature_extractor = feature_extractor.cuda()
    feature_extractor.eval()

    dataloaders, model = get_all(modes=[])
    display_config = Munch(display_config)
    model = load_ckpt(model)
    model.eval()

    for input_video in VIDEO_NAMES:
        print(f'Processing {input_video}...')
        input_video = VIDEO_DIR + input_video
        try:
            probe = ffmpeg.probe(input_video)
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            raise e
        video_stream = next((stream for stream in probe['streams'] 
                            if stream['codec_type'] == 'video'), None)
        orig_width = int(video_stream['width'])
        orig_height = int(video_stream['height'])

        width = model_config.input_resolution * 2
        height = model_config.input_resolution

        cmd = (
            ffmpeg.input(input_video).filter('scale', width, height)
        )

        try:
            out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )
        except ffmpeg.Error as e:
            # print(e.stderr.decode())
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            raise e
        
        video = np.frombuffer(out, np.uint8)
        video = video.reshape([-1, height, width, 3])

        if DATASET_NAME == 'Wu_MMSys_17':
            vid = str(int(input_video.split('/')[-1].split('.')[0].split('-')[-1]) - 1)
            fps = eval(video_stream['r_frame_rate'])
        elif DATASET_NAME == 'David_MMSys_18':
            vid = input_video.split('/')[-1]
            fps_file = open(VIDEO_DIR + 'fps', 'r')
            # fps_file中, 每个视频的fps信息占一行, 先是视频名, 然后是fps, 以空格分隔;
            fps = float(fps_file.readlines()[int(vid.split('_')[0])-1].split(' ')[1])
            fps_file.close()
            print(f'fps: {fps}')
        elif DATASET_NAME == 'Xu_CVPR_18':
            vid = input_video.split('/')[-1].split('.')[0]
            fps = eval(video_stream['r_frame_rate'])
        elif DATASET_NAME == 'Nasrabadi_MMSys_19':
            vid = input_video.split('/')[-1]
            fps = eval(video_stream['r_frame_rate'])
        else:
            raise NotImplementedError

        salmaps = np.zeros((NUM_SAMPLES_PER_VIDEO[vid], *SALMAP_SHAPE)) if SALMAP_SHAPE is not None else np.zeros((NUM_SAMPLES_PER_VIDEO[vid], 224, 448))  # 因为模型的原始输出就是224x448;
        features = np.zeros((NUM_SAMPLES_PER_VIDEO[vid], 393, 768))
        for i in tqdm(range(NUM_SAMPLES_PER_VIDEO[vid])):
            fi = i * fps * SAMPLING_RATE
            fi = min(max(int(fi + 0.5), 2), video.shape[0]-3)  # 将fi四舍五入到最近的整数, 并限制在[2, video.shape[0]-3]之间;
            frames = video[fi-2:fi+3]
            frames = torch.from_numpy(frames.astype('float32')).permute(0, 3, 1, 2)
            frames = ((frames / 255.) - 0.5) / 0.5
            frames = frames.cuda()
            
            encoded_feat = feature_extractor(frames)
            features[i] = encoded_feat.detach().cpu().numpy()[2]

            frame_ = encoded_feat[:, 1:].unsqueeze(0)  # TNC -> BTNC
            cls_ = encoded_feat[:, 0].unsqueeze(0)  # TC -> BTC
            mask_ = torch.ones(1, num_frame).cuda()  # BT

            result = model({'frame': frame_, 'cls': cls_}, {'mask': mask_})  # dict_keys(['loss_total', 'cls_weight', 'loss_cls', 'loss_feat_spatial', 'loss_feat_temporal', 'output'])
            result['heatmap'] = model.compute_heatmap(result['output'].contiguous())
            img = result['heatmap'].cpu().numpy()[0][2]
            if SALMAP_SHAPE is not None:
                img = cv2.resize(img, dsize=(SALMAP_SHAPE[1], SALMAP_SHAPE[0]), interpolation=cv2.INTER_AREA)
            salmaps[i] = img

        np.save(f'{OUTPUT_DIR}{vid.split(".")[0]}.npy', salmaps)
        # np.save(f'{OUTPUT_DIR}{vid}_feat.npy', features)
    return 0


    video = torch.from_numpy(video.astype('float32')).permute(0, 3, 1, 2)[2000:2000+num_frame]
    video = ((video / 255.) - 0.5) / 0.5 # Google inception mean & std
    feature_extractor = get_extractor()

    feature_extractor = feature_extractor.cuda()    # self.model.cuda() throws error!
    feature_extractor.eval()

    encoded_feat = feature_extractor(video.cuda())

    iid = input_video.split('/')[-1][:-4]
    torch.save(encoded_feat, f'{OUTPUT_DIR}{iid}_feat.pt')

    del feature_extractor

    # Run inference
    dataloaders, model = get_all(modes=[])
    display_config = Munch(display_config)

    model = load_ckpt(model)

    model.eval()

    val_split = 'val' if 'val' in dataloaders.keys() else 'test'

    # result = model({'frame': encoded_feat[:, 1:].unsqueeze(0), 'cls': encoded_feat[:, 0].unsqueeze(0)},
    #                {'mask': torch.Tensor([1., 1., 1., 1., 1.]).unsqueeze(0).cuda()})

    frame_ = encoded_feat[:, 1:].unsqueeze(0)  # TNC -> BTNC
    cls_ = encoded_feat[:, 0].unsqueeze(0)  # TC -> BTC
    mask_ = torch.ones(1, num_frame).cuda()  # BT
    print('frame', frame_.size())  # torch.Size([1, 5, 392, 768])
    print('cls', cls_.size())  # torch.Size([1, 5, 768])
    print('mask', mask_.size())  # torch.Size([1, 5])
    result = model({'frame': frame_, 'cls': cls_}, {'mask': mask_})

    result['heatmap'] = model.compute_heatmap(result['output'].contiguous())
    vis = torch.cat([visualize_heatmap(result['heatmap'][0][j], overlay=False).unsqueeze(0)
                     for j in range(num_frame)]).unsqueeze(0)

    print(vis.size())

    torchvision.io.write_video(f'{OUTPUT_DIR}{iid}_out.mp4',
                               vis.squeeze(0).permute(0,2,3,1), fps=4)

    return 0