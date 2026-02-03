import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import struct
from scipy import signal, ndimage
import re

# -------------------------- 新增：PGM(P5)解析与转换 --------------------------
def read_pgm_p5(pgm_path):
    """解析PGM P5文件：读取544字节头，后续数据区为4字节int32"""
    with open(pgm_path, 'rb') as f:
        # 1. 读取前544个字节作为文件头
        header_blob = f.read(544)
        header_text = header_blob.decode('ascii', errors='ignore')
        
        # 提取注释行（以#开头）
        comments = [l.strip() for l in header_text.splitlines() if l.strip().startswith('#')]
        
        # 提取非注释的数据行
        lines = [l.strip() for l in header_text.splitlines() if l.strip() and not l.startswith('#')]
        if not lines or not lines[0].startswith('P5'):
            raise ValueError("PGM格式错误：未在头信息中找到P5标识")
        
        # 解析尺寸和最大值
        try:
            dims = lines[1].split()
            width, height = int(dims[0]), int(dims[1])
            maxval = int(lines[2])
        except (IndexError, ValueError) as e:
            raise ValueError(f"PGM头信息解析失败（尺寸或最大值缺失）: {e}")
        
        # 2. 尝试从注释中提取 HalfAngle(rad)
        half_angle_rad = None
        for comment in comments:
            # 匹配模式：HalfAngle(rad): 0.5700
            match = re.search(r'HalfAngle\(rad\):\s*([\d\.]+)', comment)
            if match:
                half_angle_rad = float(match.group(1))
                break
        
        # 3. 寻址到 544 字节开始读取数据区
        f.seek(544)
        # 正确方式：数据点为 4 字节，读取为 int32
        data = np.fromfile(f, dtype=np.int32)
        
    # 转换为二维数组 (height, width)
    expected_size = width * height
    if data.size < expected_size:
        print(f"警告：数据区长度({data.size})小于头信息定义的尺寸({expected_size})")
        # 尽量填充
        img = np.zeros((height, width), dtype=np.int32)
        available = min(data.size, expected_size)
        img.flat[:available] = data[:available]
    else:
        img = data[:expected_size].reshape((height, width))
        
    header = {
        "width": width, 
        "height": height, 
        "maxval": maxval,
        "comments": comments,
        "half_angle_rad": half_angle_rad
    }
    return img, header

def pgm_pixels_to_int16(img, maxval):
    """
    处理原始数据。虽然读入是int32，但后续算法(如FIR)对动态范围有要求。
    这里将其转换为 int32 精度。
    """
    # 如果已经是RF原始信号(int32)，通常不需要像8位图像那样-128
    # 保持为 int32 精度
    return img.astype(np.int32)

def save_dat_from_array(arr_int32, out_path):
    """将二维int32数组写出为.dat"""
    arr_int32.tofile(out_path)
    return out_path

# -------------------------- 核心算法模块（修正版） --------------------------
def changeB2T(file_path, n_lines=None, n_samples=None):
    """读取.dat二进制文件，数据点为4字节int32"""
    with open(file_path, 'rb') as fid:
        # 更新为 int32 读取
        data = np.fromfile(fid, dtype=np.int32)
    
    if n_lines is not None and n_samples is not None:
        total = n_lines * n_samples
        if data.size < total:
            # 容错处理
            actual_lines = data.size // n_samples
            da = data[:actual_lines * n_samples].reshape((actual_lines, n_samples))
            return da
        da = data[:total].reshape((n_lines, n_samples))
        return da
    # 回退到历史默认尺寸
    if data.size >= 240 * 7618:
        da = data[:240 * 7618].reshape((240, 7618))
        return da
    raise ValueError("无法推断DAT尺寸，请提供n_lines与n_samples")

def processF(Data):
    """单条扫描线数据处理（修正降采样逻辑）"""
    N = len(Data)
    # FFT求频域峰值（对齐MATLAB索引）
    mag = np.abs(np.fft.fft(Data))
    m = mag.shape[0]
    # 找前半段最大值索引（MATLAB 1开始，Python+1对齐）
    max_val = np.max(mag[:int(m/2)])
    k = np.where(mag[:int(m/2)] == max_val)[0][0] + 1
    w = -2 * np.pi * k / N

    # 正交解调
    x = np.arange(N)
    Q = np.cos(w * x) * Data
    I = np.sin(w * x) * Data

    # FIR低通滤波
    T = 1 / 40000
    f = 100
    wn = 2 * T * f * 1.001
    n = 70
    fir_coeff = signal.firwin(n + 1, wn)  # 低通FIR滤波器

    Qf = signal.fftconvolve(Q, fir_coeff, mode='same')
    If = signal.fftconvolve(I, fir_coeff, mode='same')
    Fdata = np.sqrt(Q**2 + I**2)

    # 修正降采样逻辑（严格对齐MATLAB循环）
    Sdata = []
    index = 1
    while True:
        p = 16 * (index - 1) + 1  # MATLAB 1开始索引
        if p > N:
            break
        Sdata.append(Fdata[p-1])  # 转Python 0索引
        index += 1
    Sdata = np.array(Sdata) / 255.0

    # 对数压缩
    a = 1.5
    b = 0.02
    for i in range(len(Sdata)):
        Sdata[i] = a * np.log(Sdata[i] + 1) + b
        if Sdata[i] > 255:
            Sdata[i] = 255

    return Sdata

def frameProcess(file_path, n_lines=None, n_samples=None):
    """单帧数据处理（从DAT恢复二维数据，支持任意行列）"""
    data = changeB2T(file_path, n_lines=n_lines, n_samples=n_samples)
    data = np.abs(data)

    # 处理每条扫描线
    first_line = processF(data[0, :])
    Cdata = np.zeros((data.shape[0], len(first_line)))
    for n in range(data.shape[0]):
        Cdata[n, :] = processF(data[n, :])

    # 无参数映射，直接使用
    CData = Cdata

    # 转置（对应MATLAB'）
    CData = CData.T
    return CData

def image_reconstruct(d, half_angle_rad=None):
    """图像重建统一接口，内部调用优化后的凸阵重建算法"""
    return image_reconstruct_convex(d, half_angle_rad)

# -------------------------- GUI交互与显示模块（修复版） --------------------------
class BUSReconstructionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("B超图像重建系统")
        self.root.geometry("1600x900")
        
        # --- 开发者配置区：修改此处即可调整全局界面尺寸 ---
        UI_FONT_SIZE = 18       # 字体大小
        BTN_PADDING_XY = (20, 8) # 按钮内边距 (左右, 上下)
        # ----------------------------------------------

        self.style = ttk.Style()
        # 设置全局样式
        self.style.configure(".", font=("SimHei", UI_FONT_SIZE))
        # 专门设置按钮样式，使其更易点击
        self.style.configure("TButton", padding=BTN_PADDING_XY)

        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 存储加载的数据
        self.frame_data_list = []
        self.recon_data = None
        self.reference_image = None
        self.pgm_header = None
        self.probe_type = tk.StringVar(value="convex")  # 默认凸阵

        # 控件布局
        self._create_widgets()

    def _create_widgets(self):
        # 顶部控制面板
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # 第一行：文件上传
        file_frame = ttk.Frame(top_frame)
        file_frame.pack(side=tk.LEFT, padx=10)
        
        upload_btn = ttk.Button(file_frame, text="上传PGM数据", command=self.upload_pgm_file)
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        ref_btn = ttk.Button(file_frame, text="上传参考图像", command=self.upload_reference_image)
        ref_btn.pack(side=tk.LEFT, padx=5)

        # 第二行：参数选择与执行
        action_frame = ttk.Frame(top_frame)
        action_frame.pack(side=tk.LEFT, padx=30)
        
        ttk.Label(action_frame, text="探头类型:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(action_frame, text="凸阵", variable=self.probe_type, value="convex").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(action_frame, text="线阵", variable=self.probe_type, value="linear").pack(side=tk.LEFT, padx=2)
        
        recon_btn = ttk.Button(action_frame, text="开始重建", command=self.reconstruct_image)
        recon_btn.pack(side=tk.LEFT, padx=20)

        # 下方图像显示区域 - 只保留重建图像和参考图像对比
        self.fig = plt.figure(figsize=(28, 14))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.1)
        
        self.ax2 = self.fig.add_subplot(gs[0, 0])
        self.ax3 = self.fig.add_subplot(gs[0, 1])
        
        self.ax2.set_title("重建后图像", fontsize=20)
        self.ax3.set_title("参考图像", fontsize=20)
        
        # 强制绘图区域比例为正方形，方便对比
        self.ax2.set_box_aspect(1)
        self.ax3.set_box_aspect(1)
        
        # 隐藏坐标轴刻度但保留外框
        for ax in [self.ax2, self.ax3]:
            ax.set_xticks([])
            ax.set_yticks([])
            # 设置黑色外框
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.5)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # 调整 subplots_adjust 以减少白边，让图像占满画布
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)

    def upload_reference_image(self):
        """上传并显示参考图像"""
        ref_path = filedialog.askopenfilename(
            title="选择参考图像文件",
            filetypes=[("图像文件", "*.bmp *.jpg *.jpeg"), ("所有文件", "*.*")]
        )
        if not ref_path:
            return
        try:
            # 使用 matplotlib 读取图像
            img = plt.imread(ref_path)
            self.reference_image = img
            
            self.ax3.clear()
            self.ax3.imshow(img)
            self.ax3.set_title(f"参考图像\n({ref_path.split('/')[-1]})")
            # 重新应用边框设置（clear会清除spines设置）
            self.ax3.set_xticks([])
            self.ax3.set_yticks([])
            for spine in self.ax3.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.5)
            
            self.ax3.set_box_aspect(1)
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("错误", f"无法加载参考图像：{str(e)}")

    def upload_pgm_file(self):
        """上传PGM(P5)文件，解析头信息并转换为DAT"""
        pgm_path = filedialog.askopenfilename(
            title="选择PGM格式B超数据文件",
            filetypes=[("PGM文件", "*.pgm"), ("所有文件", "*.*")]
        )
        if not pgm_path:
            return
        try:
            img, header = read_pgm_p5(pgm_path)
            self.pgm_header = header
            # 将PGM像素居中为有符号int16
            signed = pgm_pixels_to_int16(img, header["maxval"])
            # 行作为扫描线，列为采样点
            n_lines, n_samples = signed.shape[0], signed.shape[1]
            # 写出DAT
            dat_path = pgm_path[:-4] + ".dat"
            save_dat_from_array(signed, dat_path)
            # 处理一帧
            frame_data = frameProcess(dat_path, n_lines=n_lines, n_samples=n_samples)
            self.frame_data_list = [frame_data]
            # 信息提示
            angle_info = f"\n识别到扫描角度 (HalfAngle): {header['half_angle_rad']:.4f} rad" if header.get('half_angle_rad') else "\n未识别到角度信息，将使用默认值"
            meta_info = ""
            if header["comments"]:
                meta_info = "\n".join(header["comments"][:5])
            messagebox.showinfo(
                "成功",
                f"PGM加载完成：{pgm_path}\n尺寸：{header['width']}×{header['height']}\nmaxval：{header['maxval']}\n已生成DAT：{dat_path}{angle_info}\n\n文件头摘要：\n{meta_info}"
            )
            self.show_reconstructed_placeholder()
        except Exception as e:
            messagebox.showerror("错误", f"PGM处理失败：{str(e)}")

    def show_reconstructed_placeholder(self):
        """上传后清空并准备显示"""
        self.ax2.clear()
        self.ax2.set_title("重建后图像 (待重建)")
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        for spine in self.ax2.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        self.canvas.draw()

    def reconstruct_image(self):
        """执行图像重建并显示"""
        if not self.frame_data_list:
            messagebox.showwarning("警告", "请先上传数据文件！")
            return

        # 平均后重建
        d_avg = np.mean(self.frame_data_list, axis=0)
        
        # 根据用户选择的探头类型进行重建
        if self.probe_type.get() == "linear":
            self.recon_data = image_reconstruct_linear(d_avg)
            aspect_type = 'auto'  # 线阵通常自适应比例
        else:
            # 提取头信息中的扫描角度
            half_angle = self.pgm_header.get("half_angle_rad") if self.pgm_header else None
            self.recon_data = image_reconstruct_convex(d_avg, half_angle_rad=half_angle)
            aspect_type = 'equal' # 凸阵必须等比例保证几何准确

        # 显示重建图像
        self.ax2.clear()
        img = np.ma.masked_less(self.recon_data, 0)
        cm = plt.cm.gray.copy()
        cm.set_bad(color='black')
        
        self.ax2.imshow(img, cmap=cm, origin='upper', aspect=aspect_type, 
                        interpolation='nearest', extent=[0, img.shape[1], img.shape[0], 0])
        
        # 在标题中显示角度信息
        angle_str = ""
        if self.probe_type.get() == "convex":
            angle = self.pgm_header.get("half_angle_rad") if self.pgm_header else None
            if angle:
                angle_str = f" (角度: {angle:.4f} rad)"
            else:
                angle_str = " (角度: 默认 68°)"
        
        self.ax2.set_title(f"重建后图像 ({'线阵' if self.probe_type.get() == 'linear' else '凸阵'}){angle_str}")
        
        # 重新应用黑色外框
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        for spine in self.ax2.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
            
        self.ax2.margins(0)
        # 移除可能导致警告的 tight_layout，改用 subplots_adjust 手动控制边距
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
        self.canvas.draw()

    def on_closing(self):
        """彻底结束程序进程"""
        try:
            self.root.quit()     # 停止 Tkinter 事件循环
            self.root.destroy()  # 销毁所有窗口组件
            import sys
            sys.exit(0)         # 强制退出 Python 解释器，释放终端
        except Exception:
            import sys
            sys.exit(0)

def image_reconstruct_linear(d):
    """线阵图像重建：使用双三次插值优化显示效果"""
    # 增加插值倍率，使图像更平滑，消除马赛克感
    # order=3 表示使用双三次插值 (Bicubic)
    scale_h = 2.0
    scale_w = 4.0
    recon = ndimage.zoom(d, (scale_h, scale_w), order=3)
    return recon

def image_reconstruct_convex(d, half_angle_rad=None):
    """凸阵图像重建：使用反向映射与双线性插值，彻底消除空洞点"""
    m, n = d.shape
    if half_angle_rad is None:
        total_angle_deg = 68
        half_angle_rad = (total_angle_deg / 2) / 180 * np.pi
    
    total_angle_rad = 2 * half_angle_rad
    r_start = 70.0  # 起始半径
    
    # 1. 计算重建图像所需的实际尺寸
    R_max = r_start + m
    max_width = 2 * R_max * np.sin(half_angle_rad)
    max_height = R_max - r_start * np.cos(half_angle_rad)
    
    # 可以通过 res_scale 调整输出分辨率
    res_scale = 1.0
    new_w = int(np.ceil(max_width * res_scale)) + 2
    new_h = int(np.ceil(max_height * res_scale)) + 2
    
    # 2. 建立目标网格 (Cartesian)
    xv, yv = np.meshgrid(np.arange(new_w), np.arange(new_h))
    
    # 中心偏移逻辑
    X_center = (new_w - 1) / 2.0
    Y_offset = r_start * np.cos(half_angle_rad)
    
    # 物理坐标转换 (以探头顶点为原点)
    px = xv - X_center
    py = -yv - Y_offset
    
    # 3. 物理坐标 -> 极坐标
    R = np.sqrt(px**2 + py**2)
    # theta 是相对于中心轴的角度，范围约 [-half_angle, half_angle]
    theta = np.arctan2(px, -py)
    
    # 4. 极坐标 -> 原始数据索引
    # R 范围 [r_start, r_start + m] -> 映射到 [0, m-1]
    row_idx = R - r_start
    # theta 映射到 [0, n-1]
    col_idx = (theta + half_angle_rad) / (2 * half_angle_rad) * (n - 1)
    
    # 5. 使用 ndimage.map_coordinates 进行双线性插值 (order=1)
    # order=1 为双线性，若需要更平滑可用 order=3 (双三次)
    # cval=-1 用于标识背景区域
    recon = ndimage.map_coordinates(d, [row_idx, col_idx], order=1, mode='constant', cval=-1)
    
    return recon

# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":
    # 解决中文显示问题
    plt.rcParams["font.family"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False

    root = tk.Tk()
    app = BUSReconstructionApp(root)
    root.mainloop()
