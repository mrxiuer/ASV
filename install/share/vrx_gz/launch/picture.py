import matplotlib.pyplot as plt
import os
import numpy as np

def read_ctl_file(file_path):
    """读取CTL.txt文件并解析数据"""
    indices = []
    current_headings = []
    target_headings = []

    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过表头
            parts = line.strip().split(",")
            if len(parts) >= 3:  # 确保至少有3个部分
                try:
                    idx = parts[0].strip()
                    curr = parts[1].strip()
                    targ = parts[2].strip()
                    
                    if idx and curr and targ:  # 确保所有值都不为空
                        indices.append(float(idx))
                        current_headings.append(float(curr))
                        target_headings.append(float(targ))
                except ValueError as e:
                    print(f"跳过无效行: {line.strip()} - {e}")
                    continue

    return indices, current_headings, target_headings

def plot_headings(indices, current_headings, target_headings, output_path="headings_plot.png"):
    """绘制当前偏航角和目标角度对比图"""
    plt.figure(figsize=(12, 6))
    
    # 绘制当前偏航角（蓝色实线）
    plt.plot(indices, current_headings, label="Current Angle", color="blue")
    
    # 绘制目标角度（红色虚线）
    plt.plot(indices, target_headings, label="Target Angle", linestyle="--", color="red")
    
    # 设置图表标题和标签
    plt.title("Current Angle VS Target Angle")
    plt.xlabel("Index")
    plt.ylabel("Angle (radians)")
    plt.grid(True)
    plt.legend()
    
    # 保存图像
    plt.savefig(output_path, dpi=300)
    print(f"图像已保存到: {output_path}")
    
    # 显示图像
    plt.show()

def main():
    ctl_file_path = "../../lstm_model_save/CTL.txt"
    
    # 检查文件是否存在
    if not os.path.exists(ctl_file_path):
        print(f"错误: 文件 {ctl_file_path} 不存在!")
        return
    
    # 读取数据
    indices, current_headings, target_headings = read_ctl_file(ctl_file_path)
    
    # 检查是否成功读取数据
    if len(indices) == 0:
        print("错误: 未能从文件中读取到有效数据!")
        return
    
    print(f"成功读取 {len(indices)} 个数据点")
    
    # 绘制当前偏航角和目标角度对比图
    plot_headings(indices, current_headings, target_headings, output_path="headings_plot.png")

if __name__ == "__main__":
    main()