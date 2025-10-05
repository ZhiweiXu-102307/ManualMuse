import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import csv
import os

def extract_audio_amplitude(mp3_file_path, output_csv=None, sample_rate=480):
    """
    提取MP3文件的音频幅值波动曲线
    
    参数:
    mp3_file_path: MP3文件路径
    output_csv: 输出CSV文件路径 (可选)
    sample_rate: 采样率 (每秒采样点数，默认100)
    
    返回:
    amplitudes: 幅值数组
    time_points: 时间点数组
    """
    
    # 加载音频文件
    print(f"正在加载音频文件: {mp3_file_path}")
    audio = AudioSegment.from_mp3(mp3_file_path)
    
    # 转换为单声道并获取原始数据
    audio = audio.set_channels(1)  # 转换为单声道
    samples = np.array(audio.get_array_of_samples())
    
    # 计算采样间隔
    original_sample_rate = audio.frame_rate
    print(f"原始采样率: {original_sample_rate} Hz")
    sample_interval = min(1, original_sample_rate // sample_rate)
    print(f"采样间隔: {sample_interval} 个采样点")
    
    # 降采样获取幅值数据
    print("正在处理音频数据...")
    downsampled_samples = samples[::sample_interval]
    
    # 计算时间点
    duration = len(audio) / 1000.0  # 转换为秒
    time_points = np.linspace(0, duration, len(downsampled_samples))
    
    # 归一化幅值 (-1 到 1)
    max_amplitude = np.max(np.abs(downsampled_samples))
    if max_amplitude > 0:
        amplitudes = downsampled_samples / max_amplitude
    else:
        amplitudes = downsampled_samples
    
    # 保存为CSV文件
    if output_csv:
        save_to_csv(time_points, amplitudes, output_csv)
    
    return time_points, amplitudes

def save_to_csv(time_points, amplitudes, csv_file_path):
    """将时间和幅值数据保存为CSV文件"""
    print(f"正在保存数据到: {csv_file_path}")
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time(s)', 'Amplitude'])
        for t, amp in zip(time_points, amplitudes):
            writer.writerow([t, amp])
    print(f"数据已保存到 {csv_file_path}")

def plot_amplitude_waveform(time_points, amplitudes, title="音频幅值波动曲线"):
    """绘制幅值波动曲线图"""
    print("正在生成可视化图表...")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, amplitudes, linewidth=0.5, color='blue', alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel('Time(s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 设置合适的x轴刻度
    max_time = time_points[-1]
    if max_time > 60:
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f'{int(x//60)}分{int(x%60)}秒' if x >= 60 else f'{x:.1f}秒'))
    
    plt.show()

if __name__ == "__main__":
    pass