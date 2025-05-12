import matplotlib.pyplot as plt

def read_ctl_file(file_path):
    """读取CTL.txt文件并解析数据"""
    time_stamps = []
    current_headings = []
    target_headings = []
    losses = []

    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过表头
            parts = line.strip().split(",")
            if len(parts) == 4:
                try:
                    time_stamps.append(float(parts[0]) if parts[0].strip() else None)
                    current_headings.append(float(parts[1]) if parts[1].strip() else None)
                    target_headings.append(float(parts[2]) if parts[2].strip() else None)
                    losses.append(float(parts[3]) if parts[3].strip() else None)
                except ValueError:
                    # 如果某一行数据格式不正确，跳过该行
                    print(f"跳过无效行: {line.strip()}")
                    continue

    return time_stamps, current_headings, target_headings, losses

def plot_loss(time_stamps, losses, output_path="loss_plot.png"):
    """绘制损失函数变化图像"""
    # 过滤掉 None 值
    valid_indices = [i for i, loss in enumerate(losses) if loss is not None]
    valid_losses = [losses[i] for i in valid_indices]

    plt.figure()
    plt.plot(valid_indices, valid_losses, label="Loss", color="blue")
    plt.title("Loss Function Over Data Points")
    plt.xlabel("Data Point Index")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()
    print(f"损失函数图像已保存到 {output_path}")
    
def plot_headings(time_stamps, current_headings, target_headings, output_path="headings_plot.png"):
    """绘制当前偏航角和目标角度对比图"""
    # 过滤掉 None 值
    valid_indices = [i for i, (ch, th) in enumerate(zip(current_headings, target_headings)) if ch is not None and th is not None]
    valid_current_headings = [current_headings[i] for i in valid_indices]
    valid_target_headings = [target_headings[i] for i in valid_indices]

    plt.figure()
    plt.plot(valid_indices, valid_current_headings, label="Current Heading", color="blue")
    plt.plot(valid_indices, valid_target_headings, label="Target Heading", linestyle="--", color="red")
    plt.title("Current Heading vs Target Heading")
    plt.xlabel("Data Point Index")
    plt.ylabel("Heading (radians)")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()
    print(f"偏航角对比图像已保存到 {output_path}")
    
def main():
    ctl_file_path = "../../lstm_model_save/CTL.txt"  # 替换为实际路径
    time_stamps, current_headings, target_headings, losses = read_ctl_file(ctl_file_path)

    # 绘制损失函数变化图像
    plot_loss(time_stamps, losses, output_path="loss_plot.png")

    # 绘制当前偏航角和目标角度对比图
    plot_headings(time_stamps, current_headings, target_headings, output_path="headings_plot.png")

if __name__ == "__main__":
    main()