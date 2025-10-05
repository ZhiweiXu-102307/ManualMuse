from pydub import AudioSegment
import os

def convert_m4a_to_mp3_pydub(input_m4a_path, output_mp3_path=None, bitrate="192k"):
    """
    使用pydub将M4A转换为MP3
    
    参数:
    input_m4a_path: 输入M4A文件路径
    output_mp3_path: 输出MP3文件路径（可选）
    bitrate: MP3比特率，默认"192k"
    """
    try:
        # 如果没有指定输出路径，自动生成
        if output_mp3_path is None:
            base_name = os.path.splitext(input_m4a_path)[0]
            output_mp3_path = f"{base_name}.mp3"
        
        print(f"正在转换: {input_m4a_path}")
        
        # 加载M4A文件
        audio = AudioSegment.from_file(input_m4a_path, format="m4a")
        
        # 导出为MP3
        audio.export(output_mp3_path, format="mp3", bitrate=bitrate)
        
        print(f"转换完成: {output_mp3_path}")
        print(f"文件信息: {len(audio)/1000:.1f}秒, {audio.frame_rate}Hz, {audio.channels}声道")
        
        return output_mp3_path
        
    except Exception as e:
        print(f"转换失败: {str(e)}")
        return None
    
if __name__ == "__main__":
    # 示例: 将M4A文件转换为MP3
    input_m4a_path = "./MP3/60.m4a"
    output_mp3_path = convert_m4a_to_mp3_pydub(input_m4a_path)