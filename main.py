import os
from utils.to_data import extract_audio_amplitude, save_to_csv
from utils.transform_to import transform

# 确保目录存在
os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)

def process_mp3_files():
    """
    批量处理 MP3 文件，提取幅值数据并保存为 CSV 文件
    """
    mp3_dir = "MP3"
    data_dir = "data"
    
    # 遍历 MP3 文件夹中的所有 MP3 文件
    for mp3_file in os.listdir(mp3_dir):
        if mp3_file.endswith(".mp3"):
            mp3_path = os.path.join(mp3_dir, mp3_file)
            csv_file = os.path.join(data_dir, f"{os.path.splitext(mp3_file)[0]}_amplitude.csv")
            
            print(f"正在处理文件: {mp3_file}")
            
            # 提取幅值数据并保存为 CSV
            try:
                time_points, amplitudes = extract_audio_amplitude(mp3_path)
                save_to_csv(time_points, amplitudes, csv_file)
                print(f"数据已保存到: {csv_file}")
            except Exception as e:
                print(f"处理文件 {mp3_file} 时出错: {e}")

def process_data_files():
    """
    批量处理 data 文件夹中的 CSV 文件，进行傅里叶变换并保存结果
    """
    data_dir = "data"
    output_dir = "output"
    
    # 遍历 data 文件夹中的所有 CSV 文件
    for csv_file in os.listdir(data_dir):
        if csv_file.endswith("_amplitude.csv"):
            csv_path = os.path.join(data_dir, csv_file)
            output_prefix = os.path.splitext(csv_file)[0]
            
            print(f"正在处理数据文件: {csv_file}")
            
            # 进行傅里叶变换并保存结果
            try:
                transform(csv_path)
                print(f"处理完成: {csv_file}")
            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")

if __name__ == "__main__":
    print("开始处理 MP3 文件...")
    process_mp3_files()
    
    print("\n开始处理数据文件...")
    process_data_files()
    
    print("\n所有文件处理完成！")