import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# 读取CSV文件
def read_vibration_data(file_path):
    """
    读取CSV文件，第一列为时间，第二列为幅值
    跳过第一行（列名等信息）
    """
    data = pd.read_csv(file_path, header=None, skiprows=1)
    time = data.iloc[:, 0].values
    amplitude = data.iloc[:, 1].values
    return time, amplitude

# 进行傅里叶变换
def perform_fft(time, amplitude):
    """
    对振动数据进行傅里叶变换
    """
    # 采样频率
    fs = 1 / (time[1] - time[0])
    
    # 数据点数
    N = len(amplitude)
    
    # 进行FFT
    fft_values = fft(amplitude)
    
    # 计算频率轴
    freqs = fftfreq(N, 1/fs)
    
    # 取正频率部分
    positive_freqs = freqs[:N//2]
    positive_fft = 2.0/N * np.abs(fft_values[0:N//2])
    
    return positive_freqs, positive_fft, fs

# 切片傅里叶变换（STFT）
def perform_stft(time, amplitude, tt=100e-3, Nfft=4800):
    """
    对振动数据进行切片傅里叶变换
    """
    # 采样频率
    fs = 1 / (time[1] - time[0])
    
    # 计算每个时间窗口的长度（采样点数）
    Len = int(tt * fs)
    
    # 计算可以分成多少个完整的时间窗口
    N = len(amplitude) // Len
    
    # 初始化时频矩阵
    Lofar = np.zeros((N, Nfft//2))
    time_windows = np.zeros(N)
    
    for ii in range(N):
        # 提取当前时间窗口的数据
        start_idx = ii * Len
        end_idx = (ii + 1) * Len
        s = amplitude[start_idx:end_idx]
        
        # 计算FFT并取正频率部分
        fft_values = fft(s, Nfft)
        positive_fft = 2.0/Len * np.abs(fft_values[:Nfft//2])
        
        Lofar[ii, :] = positive_fft
        time_windows[ii] = time[start_idx]  # 记录每个窗口的起始时间
    
    # 计算频率轴
    freqs = fftfreq(Nfft, 1/fs)[:Nfft//2]
    
    return Lofar, time_windows, freqs, fs

# 寻找前三阶卓越频率
def find_top_three_peaks(freqs, fft_amplitude):
    """
    寻找频谱中的前三阶卓越频率
    """
    # 寻找峰值
    peaks, properties = find_peaks(fft_amplitude, height=0.1*np.max(fft_amplitude))
    
    # 如果没有找到足够的峰值，返回空值
    if len(peaks) < 3:
        print(f"警告: 只找到 {len(peaks)} 个峰值，需要至少3个")
        # 返回找到的所有峰值，不足的用NaN填充
        top_three_freqs = np.full(3, np.nan)
        top_three_amps = np.full(3, np.nan)
        peak_indices = np.full(3, -1)
        
        for i in range(min(3, len(peaks))):
            top_three_freqs[i] = freqs[peaks[i]]
            top_three_amps[i] = fft_amplitude[peaks[i]]
            peak_indices[i] = peaks[i]
            
        return top_three_freqs, top_three_amps, peak_indices
    
    # 按幅值大小排序
    peak_freqs = freqs[peaks]
    peak_amplitudes = fft_amplitude[peaks]
    
    # 按幅值降序排列
    sorted_indices = np.argsort(peak_amplitudes)[::-1]
    
    # 获取前三阶卓越频率
    top_three_freqs = peak_freqs[sorted_indices[:3]]
    top_three_amps = peak_amplitudes[sorted_indices[:3]]
    
    return top_three_freqs, top_three_amps, peaks[sorted_indices[:3]]

# 在时频图中标注峰值频率
def mark_peak_frequencies_in_stft(ax, stft_matrix, stft_times, stft_freqs, max_peaks_per_slice=3):
    """
    在时频图中用白色圆点标注每个时间切片的频率峰值
    """
    # 限制显示的峰值数量，避免过于拥挤
    display_interval = max(1, len(stft_times) // 20)  # 最多显示20个时间点的峰值
    
    for i, time_idx in enumerate(range(0, len(stft_times), display_interval)):
        # 获取当前时间切片的频谱
        spectrum = stft_matrix[time_idx, :]
        
        # 寻找峰值
        peaks, properties = find_peaks(spectrum, height=0.1*np.max(spectrum))
        
        if len(peaks) > 0:
    # 按幅值排序，取前一个峰值（最大值）
            peak_amplitudes = spectrum[peaks]
            sorted_indices = np.argsort(peak_amplitudes)[::-1][:1]  # 只取前1个峰值
    
            for idx in sorted_indices:
                peak_freq = stft_freqs[peaks[idx]]
                peak_amp = peak_amplitudes[idx]
        
            # 绘制白色圆点
                ax.plot(stft_times[time_idx], peak_freq, 'wo', markersize=4, 
                    markeredgecolor='white', markeredgewidth=0.5)
                
                # 标注频率值（每隔几个点标注一次，避免过于拥挤）
                if i % 3 == 0:  # 每3个显示点标注一次
                    ax.text(stft_times[time_idx] + 0.02 * (stft_times[-1] - stft_times[0]), 
                           peak_freq, f'{peak_freq:.0f} Hz', 
                           color='white', fontsize=12, fontweight='bold')

# 可视化结果
def plot_results(time, amplitude, freqs, fft_amplitude, top_freqs, top_amps, 
                peak_indices, stft_matrix, stft_times, stft_freqs, file_path=None):
    """
    绘制时域波形、频域频谱图和时频图
    """
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams.update({
        'font.family': ['SimSun', 'Times New Roman'],
        'font.size': 16,
        'font.weight': 'bold',
        'axes.labelsize': 18,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.fontsize': 14,
        'legend.frameon': False,
        'lines.linewidth': 1,
        'lines.markersize': 6,
        'axes.linewidth': 3,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
    })

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 时域波形
    axes[0].plot(time, amplitude, 'b-')
    axes[0].set_xlim(0, np.max(time))
    axes[0].set_xlabel('时间 (s)')
    axes[0].set_ylabel('幅值')
    axes[0].set_title('时域波形')
    axes[0].grid(False)
    
    # 频域频谱
    axes[1].plot(freqs, fft_amplitude, 'r-')
    axes[1].set_xlim(0, 10000)
    axes[1].set_xlabel('频率 (Hz)')
    axes[1].set_ylabel('幅值')
    axes[1].set_title('频域频谱')
    axes[1].grid(False)
    
    # 标记前三阶卓越频率
    colors = ['green', 'blue', 'purple']
    labels = ['一阶', '二阶', '三阶']
    
    valid_peaks = 0
    for i, (freq, amp, idx) in enumerate(zip(top_freqs, top_amps, peak_indices)):
        if not np.isnan(freq) and idx != -1:
            axes[1].plot(freq, amp, 'o', color=colors[i], markersize=6, 
                        label=f'{labels[i]}频率: {freq:.2f} Hz')
            valid_peaks += 1
    
    if valid_peaks > 0:
        axes[1].legend()
    
    # 时频图（以颜色深浅表示频率幅值）
    im = axes[2].imshow(stft_matrix.T, aspect='auto', origin='lower',
                       extent=[stft_times[0], stft_times[-1], stft_freqs[0], stft_freqs[-1]],
                       cmap='viridis')
    axes[2].set_xlabel('时间 (s)')
    axes[2].set_ylabel('频率 (Hz)')
    axes[2].set_title('时频分析图')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=axes[2])
    cbar.set_label('幅值')
    
    # 设置频率范围
    axes[2].set_ylim(0, min(10000, np.max(stft_freqs)))
    
    # 在时频图中标注每个时间切片的峰值频率
    mark_peak_frequencies_in_stft(axes[2], stft_matrix, stft_times, stft_freqs)
    
    plt.tight_layout()
    
    # 保存图形到 output 文件夹
    if file_path:
        os.makedirs("output", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join("output", f"{base_name}_plot.png")
        plt.savefig(output_path)
        print(f"图形已保存到: {output_path}")
    
    plt.show()
    
    # 打印卓越频率信息
    print("前三阶卓越频率:")
    for i, (freq, amp) in enumerate(zip(top_freqs, top_amps)):
        if not np.isnan(freq):
            print(f"{i+1}阶: {freq:.4f} Hz, 幅值: {amp:.4f}")
        else:
            print(f"{i+1}阶: 未找到有效峰值")

# 主函数
def transform(file_path, tt=100e-3, Nfft=4800):
    """
    主处理函数
    """
    # 读取数据
    time, amplitude = read_vibration_data(file_path)
    print(f"读取到 {len(time)} 个数据点")
    print(f"时间范围: {time[0]:.4f} - {time[-1]:.4f} s")
    print(f"采样频率: {1/(time[1]-time[0]):.2f} Hz")
    
    # 进行傅里叶变换
    freqs, fft_amplitude, fs = perform_fft(time, amplitude)
    print(f"频率分辨率: {freqs[1]-freqs[0]:.4f} Hz")
    
    # 进行切片傅里叶变换
    stft_matrix, stft_times, stft_freqs, fs_stft = perform_stft(time, amplitude, tt, Nfft)
    print(f"STFT时间窗口数: {len(stft_times)}, 频率点数: {len(stft_freqs)}")
    
    # 寻找前三阶卓越频率
    top_freqs, top_amps, peak_indices = find_top_three_peaks(freqs, fft_amplitude)
    
    # 可视化结果
    plot_results(time, amplitude, freqs, fft_amplitude, top_freqs, top_amps, 
                peak_indices, stft_matrix, stft_times, stft_freqs, file_path)
    
    # 保存时频数据到Excel
    save_stft_to_excel(stft_matrix, stft_times, stft_freqs, file_path)
    
    return top_freqs, top_amps, stft_matrix

def save_stft_to_excel(stft_matrix, stft_times, stft_freqs, file_path):
    """
    保存时频数据到Excel文件
    """
    try:
        # 创建DataFrame，行为时间，列为频率
        df = pd.DataFrame(stft_matrix, index=stft_times, columns=stft_freqs)
        
        # 保存到Excel
        os.makedirs("output", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        excel_path = os.path.join("output", f"{base_name}_频域表.xlsx")
        
        df.to_excel(excel_path)
        print(f"时频数据已保存到: {excel_path}")
        
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")

# 使用示例
if __name__ == "__main__":
    # 示例用法
    # file_path = "your_vibration_data.csv"  # 替换为您的文件路径
    # top_freqs, top_amps, stft_matrix = transform(file_path)

    pass