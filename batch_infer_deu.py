import os
import numpy as np
import soundfile as sf
from datasets import load_dataset
import subprocess
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm


# 第一步：加载数据集并生成WAV文件
def generate_wav_files(data_files, wav_output_dir):
    # 加载数据集
    dataset = load_dataset("parquet", data_files=data_files)

    # 确保输出目录存在
    os.makedirs(wav_output_dir, exist_ok=True)

    # 遍历数据集中的每一行
    for idx, example in enumerate(dataset['train']):
        # 获取音频数据和采样率
        audio_array = example['audio']['array']
        sampling_rate = example['audio']['sampling_rate']

        # 确保音频数据是numpy数组
        audio_np = np.array(audio_array)

        # 指定输出wav文件的路径
        output_path = os.path.join(wav_output_dir, f"audio_sample_{idx}.wav")

        # 保存为wav文件
        # sf.write(output_path, audio_np, sampling_rate)

        # print(f"WAV file {output_path} saved successfully.")


# 第二步：对每个WAV文件进行推理，生成RTTM文件，并立即计算DER
def infer_rttms_and_calculate_der(wav_dir, output_dir, yaml_config, data_files):
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    dataset = load_dataset("parquet", data_files=data_files)

    # 创建DER计算器
    der_metric = DiarizationErrorRate()

    overall_der = 0.0
    n_samples = dataset["train"].num_rows

    for i in range(n_samples):
        wav_name = f"audio_sample_{i}"

        # 进行推理
        subprocess.run([
            "python",
            "diaper/infer_single_file.py",
            "-c",
            yaml_config,
            "--wav-dir",
            wav_dir,
            "--wav-name",
            wav_name
        ])
        print(f"RTTM file for {wav_name} generated.")

        # 载入当前样本的数据
        sample = dataset["train"][i]
        timestamps_start = sample["timestamps_start"]
        timestamps_end = sample["timestamps_end"]
        speakers = sample["speakers"]

        # 构建参考Annotation对象
        reference = Annotation()
        for start, end, speaker in zip(timestamps_start, timestamps_end, speakers):
            reference[Segment(start, end)] = speaker

        # 加载推理生成的RTTM文件
        hypothesis_rttm = os.path.join(output_dir, f"{wav_name}.rttm")

        # 检查RTTM文件是否存在，且不是空
        if os.path.exists(hypothesis_rttm):
            rttm_data = load_rttm(hypothesis_rttm)
            if rttm_data:  # 如果rttm_data不为空
                hypothesis = next(iter(rttm_data.values()))
                # 计算当前样本的DER
                der = der_metric(reference, hypothesis)
            else:
                # RTTM文件为空，设置DER为100%
                print(f"Sample {i}: RTTM file is empty, assigning DER = 1.0 (100%)")
                der = 1.0
        else:
            # RTTM文件不存在，设置DER为100%
            print(f"Sample {i}: RTTM file does not exist, assigning DER = 1.0 (100%)")
            der = 1.0

        print(f"Sample {i} DER: {der:.4f}")

        # 累计DER
        overall_der += der

    # 计算整体DER
    overall_der /= n_samples
    print(f"Overall DER: {overall_der:.4f}")

    return overall_der


# 主程序执行
if __name__ == "__main__":
    # 定义文件和目录路径
    data_files = [f"CALLHOME/deu/data-0000{i}-of-00005.parquet" for i in range(5)]
    wav_output_dir = "diaper/wav_files"
    rttm_output_dir = "diaper/rttms/deu"
    yaml_config = "examples/infer_16k_10attractors.yaml"

    # 第一步：生成WAV文件
    generate_wav_files(data_files, wav_output_dir)

    # 第二步：推理每个WAV文件，生成RTTM文件，并立即计算DER
    overall_der = infer_rttms_and_calculate_der(wav_dir=wav_output_dir, output_dir=rttm_output_dir,
                                                yaml_config=yaml_config, data_files=data_files)

    print(f"Final Overall DER: {overall_der:.4f}")
