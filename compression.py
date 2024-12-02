# %%
import subprocess
import math
import os
import numpy as np
import cv2
import torch
import concurrent.futures
import traceback
import time
import threading
import pandas as pd
import matplotlib.pyplot as plt
from rppg_toolbox.src.common.import_tqdm import  tqdm
from rppg_toolbox.src.common.cache import CacheType
from rppg_toolbox.src.loss import Neg_PearsonLoss
from rppg_toolbox.src.common.cuda_info import get_device
from rppg_toolbox.src.data_generator.PhysNet import PhysNetDataConfig, PhysNetDataGenerator
from rppg_toolbox.src.dataset_reader.UBFC_Phys import UBFCPhysDatasetReader
from rppg_toolbox.src.dataset_reader.ZJXU_MOTION import ZJXU_MOTION_Reader
from rppg_toolbox.src.face_detector.mtcnn.detector import detect_faces
NUM_WORKER = 1
ffmpeg_path = os.path.expanduser(r'/usr/local/bin/ffmpeg')
root_compress_output_path = os.path.expanduser(r'~/cache/compression')
root_test_cache_path = os.path.expanduser(r"~/cache")
root_out_path = r'./out/compression'
method = "PhysNet"
# method = "POS"
TEST_CACHE = CacheType.NEW_CACHE

dataset_name = "ZJXU-MOTION"
dataset_path = os.path.expanduser(r'/public/share/weiyuanwang/dataset/ZJXU-MOTION')
# dataset_name = "UBFC-Phys"
# dataset_path = os.path.expanduser(r"/public/share/weiyuanwang/dataset/UBFC-Phys")

motion = "static"
# motion = "wark"

# codecs = ['libx264_all_i','libx265_all_i']
# suffixs = ['avi','mp4']
# compression_strengths = [1,4,8,10,14,20,24,30]
# modes = ['qp']

# codecs = ['prores_ks']
# suffixs = ['mov']
# compression_strengths = range(0, 5)
# modes = ['q']

# codecs = ['mjpeg']
# suffixs = ['avi']
# compression_strengths = range(2, 32)
# modes = ['q']

codecs = ['libx264','libx265','av1']
suffixs = ['avi','mp4','mkv']
compression_strengths = range(1, 36)
modes = ['crf','qp','b']

# codecs = ['vp9']
# suffixs = ['mkv']
# compression_strengths = range(1, 36)
# modes = ['crf','b']

# %% 
if dataset_name == "ZJXU-MOTION":
    test_dataset_reader = ZJXU_MOTION_Reader(dataset_path,samples=['s10','s18','s19'],scenes=['W_L1'] if motion == "wark" else ['S_L1'])
    list_of_info_data,list_of_video_path = test_dataset_reader.read() if TEST_CACHE == CacheType.NEW_CACHE else None
    list_of_bvp = [i.BVP for i in list_of_info_data]
    list_of_video = [i.usb_camera_left for i in list_of_video_path]
elif dataset_name == "UBFC-Phys":
    test_dataset_reader = UBFCPhysDatasetReader(dataset_path=dataset_path,dataset=2 if motion == "wark" else 1,dataset_list=['s1', 's2', 's3'])
    list_of_bvp,list_of_video = test_dataset_reader.read() if TEST_CACHE == CacheType.NEW_CACHE else None
    pass

# %%
compress_output_path = f'{root_compress_output_path}/COMPRESS_{dataset_name}'
out_path = f'{root_out_path}/{dataset_name}'
test_cache_path = f"{root_test_cache_path}/{method}/{dataset_name}/test"

os.makedirs(compress_output_path, exist_ok=True)
os.makedirs(out_path, exist_ok=True)

# shutil.rmtree(root_compress_output_path,ignore_errors=True)
# 创建互斥锁
save_psnr_to_sheet_lock = threading.Lock()
save_ssim_to_sheet_lock = threading.Lock()
save_compression_ratio_to_sheet_lock = threading.Lock()
save_compression_time_to_sheet_lock = threading.Lock()
save_pearson_and_snr_lock = threading.Lock()
draw_lock = threading.Lock()
test_dataloader_lock = threading.Lock()
print_lock = threading.Lock()

def calculate(codec,suffix, mode,compression_strength):
    name = f'{codec}_{mode}_{str(compression_strength)}'
    if codec != 'raw':
        codec_video_paths = [os.path.abspath(f'{compress_output_path}/{codec}_{motion}/{mode}_{str(compression_strength)}_{str(i)}.{suffix}') for i in  range(len(list_of_video))]
    else:
        codec_video_paths = list_of_video
    '''
        compress
    '''
    if codec != 'raw':
        compress(list_of_video,codec_video_paths,codec,mode, compression_strength)
        pass
    '''
        compression_ratios
    '''
    # calculate_compression_ratio(list_of_video,codec_video_paths,name)
    '''
        video psnr, ssim
    '''
    # calculate_video_metrics(list_of_video,codec_video_paths,codec,mode,compression_strength)

    '''
        pearson and snr
    '''
    # calculate_pearson_and_snr(codec_video_paths,list_of_bvp,name)



def compress(video_paths,output_video_paths,codec,mode, compression_strength):
    compression_times = []
    progress_bar = tqdm(video_paths, desc=f"compress: {codec} {mode} {compression_strength}")
    for i,video_path in enumerate(progress_bar):
        output_video_path = output_video_paths[i]
        bitrate = get_video_bitrate(video_path)
        dir = os.path.dirname(output_video_path)
        os.makedirs(dir, exist_ok=True)
        ffmpeg_codec = codec
        # yuv444p
        ffmpeg_command = [ffmpeg_path,'-loglevel', 'quiet', '-hide_banner',"-i", video_path,"-gpu","0"]
        if codec != 'mjpeg':
            ffmpeg_command += ["-pix_fmt", "yuv444p"]
        if mode == 'b':
            b = int(bitrate - math.sqrt(compression_strength/35) * (bitrate-100000))
            b = min(b,bitrate)
            maxrate = max(int((bitrate + b)/2),b+100000)
        if codec == 'libx265' or codec == 'libx264':
            ffmpeg_command += ["-refs","3"]
            if codec == 'libx265' and compression_strength == 0:
                ffmpeg_command += ["-x265-params","lossless=1"]
            elif  mode == 'crf' or mode == 'qp':
                ffmpeg_command += [f"-{mode}",compression_strength]
            elif mode == 'b':
                ffmpeg_command += ["-b:v",b,"-maxrate",maxrate]
            ffmpeg_command += ["-tune","psnr"]
        elif codec == 'libx265_all_i' or codec == 'libx264_all_i':
            ffmpeg_codec = 'libx265' if codec == 'libx265_all_i' else 'libx264'
            # params = "log-level=-1"
            ffmpeg_command += ["-g",1]
            if codec == 'libx265_all_i' and compression_strength == 0:
                ffmpeg_command += ["-x265-params","lossless=1"]
            elif  mode == 'crf' or mode == 'qp':
                ffmpeg_command += [f"-{mode}",compression_strength]
            elif mode == 'b':
                ffmpeg_command += ["-b:v",b,"-maxrate",maxrate]
        elif codec == 'av1':
            if compression_strength == 0:
                ffmpeg_codec = 'libaom-av1'
                ffmpeg_command += ["-crf",compression_strength,"-aom-params","lossless=1"]
            elif  mode == 'crf' or mode == 'qp':
                ffmpeg_codec = 'libsvtav1'
                ffmpeg_command += ["-svtav1-params","tune=1:preset=3","-g",5,f"-{mode}",compression_strength]
            elif mode == 'b':
                ffmpeg_codec = 'libsvtav1'
                # max 100000k 
                bitrate = 90000000
                b = int(bitrate - math.sqrt(compression_strength/35) * (bitrate-100000))
                ffmpeg_command += ["-svtav1-params","rc=1:tune=1:preset=3","-b:v",b,"-g",3]
        elif codec == 'vp9_all_i':
            ffmpeg_codec = 'libvpx-vp9'
            ffmpeg_command += ["-tune",0,"-g",1]
            if compression_strength == 0:
                ffmpeg_command += ["-lossless",1]
            elif  mode == 'crf' or mode == 'qp':
                ffmpeg_command += [f"-{mode}",compression_strength]
            elif mode == 'b':
                ffmpeg_command += ["-b:v",b,"-maxrate",maxrate]
        elif codec == 'vp9':
            ffmpeg_codec = 'libvpx-vp9'
            ffmpeg_command += ["-tune",0,"-g",3]
            if compression_strength == 0:
                ffmpeg_command += ["-lossless",1]
            elif  mode == 'crf' or mode == 'qp':
                ffmpeg_command += [f"-{mode}",compression_strength]
            elif mode == 'b':
                ffmpeg_command += ["-b:v",b,"-maxrate",maxrate]
        elif codec == 'mjpeg':
            if mode == 'q':
                ffmpeg_command += ["-q:v",compression_strength]
            elif mode == 'b':
                ffmpeg_command += ["-b:v",b,"-maxrate",maxrate]
        elif codec == 'prores_ks':
            ffmpeg_command += ["-profile:v",compression_strength]
        else:
            raise Exception("不支持的编码")
        ffmpeg_command += ["-codec",ffmpeg_codec,output_video_path]

        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        ffmpeg_command = [str(p) for p in ffmpeg_command]


        start_time = time.perf_counter()
        subprocess.run(ffmpeg_command,check=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        compression_times.append([execution_time])


    
    sheet_name = f'{codec}_{mode}_{str(compression_strength)}'
    compression_times = np.array(compression_times).T
    with save_compression_time_to_sheet_lock:
        save_to_sheet(f'compression_time',sheet_name,compression_times)
def calculate_compression_ratio(video_paths, codec_video_paths,sheet_name):
    compression_ratios = []
    video_bitrates = []
    for index in range(len(video_paths)):
        video_path = video_paths[index]
        codec_video_path = codec_video_paths[index]
        original_size = os.path.getsize(video_path)
        codec_size = os.path.getsize(codec_video_path)
        ratio = codec_size / original_size
        compression_ratios.append([ratio])
        video_bitrates.append([get_video_bitrate(codec_video_path)])
    compression_ratios = np.array(compression_ratios).T
    video_bitrates = np.array(video_bitrates).T
    # 保存
    with save_compression_ratio_to_sheet_lock:
        save_to_sheet(f'compression_ratios',sheet_name,compression_ratios)
        save_to_sheet(f'video_bitrates',sheet_name,video_bitrates)

    
def calculate_video_metrics(video_paths,codec_video_paths,codec,mode,intensity):
    videos_psnrs = []
    videos_ssims = []
    video_bitrates = []
    progress_bar = tqdm(video_paths, desc=f"video_metrics: {codec} {mode} {compression_strength}")
    for i,video_path in enumerate(progress_bar):
        codec_video_path = codec_video_paths[i]
        # break
        psnrs = []
        ssims = []
        cap1 = cv2.VideoCapture(video_path)
        cap2 = cv2.VideoCapture(codec_video_path)
        index = 0
        sleep_time = 0
        face = None
        while True:
            ret1, frame_1 = cap1.read()
            ret2, frame_2 = cap2.read()
            if not ret1 or not ret2:
                break
            if index >= 1 and index < 10:
                save_frame_diff(frame_1,frame_2,f'{out_path}/img/diff/{codec}_{motion}/{mode}__{str(intensity)}_{str(index)}.png')
            
            bounding_boxes, landmarks = detect_faces(cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB))
            if len(bounding_boxes) <= 0:
                continue
            largest_face = max(bounding_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
            x1, y1, x2, y2,score = largest_face
            x1, y1, x2, y2 = [int(v) for v in [x1, y1, x2, y2]]
            frame_1_face = frame_1[y1:y2, x1:x2]
            frame_2_face = frame_2[y1:y2, x1:x2]
            psnr = PSNR(frame_1_face,frame_2_face)
            psnrs.append(psnr)
            # ssim = SSIM(frame_1, frame_2)
            # ssims.append(ssim)
            index += 1
        videos_psnrs.append(psnrs)
        videos_ssims.append(ssims)
        cap1.release()
        cap2.release()
        if sleep_time >= 120:
            raise Exception(f'无法加载视频：{codec_video_paths[index]}')
    videos_psnrs = np.array(videos_psnrs).T
    videos_ssims = np.array(videos_ssims).T
    # 保存
    sheet_name = f'{codec}_{mode}_{str(intensity)}'
    with save_psnr_to_sheet_lock:
        save_to_sheet('psnr',sheet_name,videos_psnrs)
    with save_ssim_to_sheet_lock:
        save_to_sheet('ssim',sheet_name,videos_ssims)
    pass

def calculate_pearson_and_snr(codec_video_paths,ppgs,name):
    codec_video_paths = np.array(codec_video_paths)
    ppgs = np.array(ppgs)
    is_need_train = method != "POS"
    device = get_device()
    from PhysNet_Train import \
    T,WIDTH,HEIGHT,BATCH,sbs_physnet as sbs
    cache_root = os.path.join(test_cache_path,name)
    test_dataset_config = PhysNetDataConfig(
        cache_root=cache_root,
        cache_type=TEST_CACHE,batch_size=BATCH,
        step=T,width=WIDTH,height=HEIGHT,slice_interval=T,
        num_workers=12,generate_num_workers=12,
        discard_front = 35,discard_back = 105
    )
    test_dataset_generator = PhysNetDataGenerator(config=test_dataset_config)
    test_dataloader = test_dataset_generator.generate((codec_video_paths,ppgs))

    if is_need_train:
        model = sbs.model
        model.eval()
        model.to(device)
    else:
        from rppg_toolbox.src.model.POS import POS
        model = POS()
        model.eval()

    ploss = Neg_PearsonLoss()
    pearsons = []
    snrs = []
    with torch.no_grad():
        device = get_device()
        for i, (X, y) in enumerate(test_dataloader):
            x_tensor = torch.as_tensor(X).float().to(device)
            y_tensor = torch.as_tensor(y).float().to(device)
            y_pred = model(x_tensor)
            p = 1 - ploss(y_tensor,y_pred).item()
            s = RMSE(y_tensor,y_pred)
            pearsons.append(p)
            snrs.append(s)
            bvp_image_path = f'{out_path}/img/bvp_{motion}/{method}/{name}'
            os.makedirs(bvp_image_path, exist_ok=True)
            with draw_lock:
                for j in range(y_tensor.shape[0]):
                    draw(f'{bvp_image_path}/{str(i)}_{str(j)}.png',y_tensor[j],y_pred[j])
    datas = {
        'pearson':np.array(pearsons),
        'rmse':np.array(snrs)
    }
    with save_pearson_and_snr_lock:
        save_pearson_and_snr(name,datas)


def save_frame_diff(frame_1,frame_2,file_path):
    subtracted_frame = cv2.absdiff(frame_1, frame_2)
    gray_frame = cv2.cvtColor(subtracted_frame,cv2.COLOR_BGR2GRAY)
    gamma = 0.3
    gamma_corrected_frame = np.power(gray_frame / 255.0, gamma) * 255.0
    gamma_corrected_frame = np.uint8(gamma_corrected_frame)
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite(file_path,cv2.cvtColor(gamma_corrected_frame,cv2.COLOR_GRAY2BGR))

def save_to_sheet(name,sheet_name,datas):
    file_path = f'{out_path}/{name}_{motion}.xlsx'
    if not os.path.exists(file_path):
        pd.DataFrame().to_excel(file_path)
    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
        try:
            writer.book.remove(writer.book[sheet_name])
        except:
            pass
        df = pd.DataFrame(datas)
        df.to_excel(writer, index=False, sheet_name=sheet_name,header=[f'video_{str(i+1)}' for i in range(datas.shape[1])])



def save_pearson_and_snr(sheet_name,datas):
    file_path = f'{out_path}/{method}_pearson_and_snr_{motion}.xlsx'
    if not os.path.exists(file_path):
        pd.DataFrame().to_excel(file_path)
    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
        try:
            writer.book.remove(writer.book[sheet_name])
        except:
            pass
        df = pd.DataFrame(datas)
        df.to_excel(writer, index=False, sheet_name=sheet_name,header=True)

def draw(file_path,y_1,y_2):
    y_1 = y_1.detach().cpu().numpy().flatten()
    y_2 = y_2.detach().cpu().numpy().flatten()
    plt.clf()
    plt.plot(y_1)
    plt.plot(y_2)
    plt.savefig(file_path)
    # plt.show()

def get_video_bitrate(filename):
    ffprobe_command = ["ffprobe", "-i", filename, "-show_entries", "format=bit_rate", "-v", "quiet", "-of", "csv=p=0"]
    try:
        ffprobe_output = subprocess.check_output(ffprobe_command).decode("utf-8").strip()
    except Exception as e:
        return 0
    return int(ffprobe_output)

def PSNR(frame_1,frame_2):

    f_pow = np.power(frame_1 - frame_2,2)
    average_per_channel = np.mean(f_pow, axis=-1)
    mse = average_per_channel/(frame_1.shape[0] * frame_1.shape[1])
    if (np.sum(mse)/3) == 0:
        return np.inf
    return 10 * np.log10(np.power(255,2) / (np.sum(mse)/3))

def SSIM(frame_1, frame_2, K1=0.01, K2=0.03, L=255):
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mean_1 = np.mean(frame_1, axis=(0, 1))
    mean_2 = np.mean(frame_2, axis=(0, 1))
    var_1 = np.var(frame_1, axis=(0, 1))
    var_2 = np.var(frame_2, axis=(0, 1))
    covariance = np.cov(np.ravel(frame_1), np.ravel(frame_2))
    numerator = (2 * mean_1 * mean_2 + C1) * (2 * covariance + C2)
    denominator = (mean_1 ** 2 + mean_2 ** 2 + C1) * (var_1 + var_2 + C2)
    return np.mean(numerator / denominator)

def SNR(y_1,y_2):
    y_1 = y_1.detach().cpu().numpy().flatten()
    y_2 = y_2.detach().cpu().numpy().flatten()
    # 计算信号和噪声的功率
    signal_power = np.mean(y_1 ** 2)
    noise_power = np.mean((y_1 - y_2) ** 2)
    # 检查噪声功率是否为0
    if noise_power == 0:
        return float('inf')  # 返回正无穷，表示信号远远大于噪声
    # 计算信噪比
    snr = 10 * np.log10(signal_power / noise_power)
    
    return snr

def RMSE(y_1,y_2):
    y_1 = y_1.detach().cpu().numpy().flatten()
    y_2 = y_2.detach().cpu().numpy().flatten()
    return np.sqrt(((y_1 - y_2) ** 2).mean())

# %%
params = []
for codec,suffix in zip(codecs,suffixs):
    for mode in modes:
        for compression_strength in compression_strengths:
            params.append([codec,suffix,mode,compression_strength])
progress_bar = tqdm(params, desc="Task")
progress_threads = []
for i,param in enumerate(progress_bar):
    if len(progress_threads) < NUM_WORKER:
        thread = threading.Thread(target=calculate, args=param)
        progress_threads.append(thread)
        thread.start()
    # waiting
    while len(progress_threads) >= NUM_WORKER or \
        (len(progress_threads) != 0 and len(params) == i + 1):
        for thread in progress_threads:
            thread.join(0.2)
        progress_threads = list(filter(lambda p:p.is_alive(),progress_threads))

