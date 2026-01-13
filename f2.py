import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# 设置科研风格绘图参数
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'cm'  # 使用 LaTeX 风格数学字体

def create_framework_diagram():
    # 1. 创建画布 (16:9 比例)
    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')  # 关闭坐标轴

    # 2. 定义配色方案 (科研蓝灰色系)
    colors = {
        'input': '#E8F1F2',   # 淡灰蓝
        'm1':    '#D1E5F0',   # 浅蓝
        'm2':    '#92C5DE',   # 中蓝
        'm3':    '#4393C3',   # 深蓝 (字体需反白或加粗) -> 改为稍浅以便黑字阅读: #6FA8DC
        'out':   '#FDDBC7',   # 淡橙色 (强调输出)
        'border':'#2C3E50',   # 深色边框
        'text':  '#000000'
    }
    
    # 修改 m3 颜色保证文字清晰
    colors['m3'] = '#A6CEE3' 

    # 3. 定义方块位置和大小
    box_y = 1.5
    box_h = 4.5
    box_w = 2.6
    gap = 0.5
    start_x = 0.5

    # 坐标计算
    x_input = start_x
    x_m1 = x_input + box_w + gap
    x_m2 = x_m1 + box_w + gap
    x_m3 = x_m2 + box_w + gap
    x_out = x_m3 + box_w + gap

    boxes = [
        {'x': x_input, 'color': colors['input'], 'title': 'Input Data', 
         'content': 'FAA Enplanements\n(2000–2024, CS/P)\n\nMacro Variables\n+ Stringency'},
        
        {'x': x_m1, 'color': colors['m1'], 'title': 'Model I:\nBreak & Baseline', 
         'content': 'PELT Break Detection\n$\\rightarrow$ 2020–2022 Regime\n\nLinear Regression\n(2000–2019, No Stringency)\n\nOutput: Baseline $b_{a,t}$\nGap $g_{a,t}$'},
        
        {'x': x_m2, 'color': colors['m2'], 'title': 'Model II:\nHybrid Forecast', 
         'content': 'SVR Stage 1:\n$t$, Stringency $\\rightarrow \\hat{g}^{SVR}$\n\nXGBoost Stage 2:\nMacro + Lag $\\rightarrow \\hat{r}^{XGB}$\n\nOutput: $\\hat{y}_{a,2025}$'},
        
        {'x': x_m3, 'color': colors['m3'], 'title': 'Model III:\nUncertainty', 
         'content': 'Residual Bootstrap\n(Block + Stratified)\n(B=5000)\n\nOutput:\n95% PI, Rank Bands\nTop-$k$ Probability'},
        
        {'x': x_out, 'color': colors['out'], 'title': 'Final Deliverables', 
         'content': '2025 Top-30 Forecasts\n\nProbabilistic Ranking\n(Uncertainty-Aware)\n\nStrategic Insights'}
    ]

    # 4. 绘制方块和文字
    for box in boxes:
        # 阴影效果
        shadow = patches.FancyBboxPatch(
            (box['x']+0.05, box_y-0.05), box_w, box_h,
            boxstyle="round,pad=0.1",
            ec="none", fc="#D3D3D3", zorder=1
        )
        ax.add_patch(shadow)

        # 主方块
        rect = patches.FancyBboxPatch(
            (box['x'], box_y), box_w, box_h,
            boxstyle="round,pad=0.1",
            linewidth=1.5, edgecolor=colors['border'], facecolor=box['color'], zorder=2
        )
        ax.add_patch(rect)

        # 标题 (加粗)
        ax.text(box['x'] + box_w/2, box_y + box_h - 0.4, box['title'],
                ha='center', va='top', fontsize=12, fontweight='bold', color='#2C3E50', zorder=3)
        
        # 分割线
        ax.plot([box['x'], box['x'] + box_w], [box_y + box_h - 1.0, box_y + box_h - 1.0], 
                color=colors['border'], lw=1, zorder=3)

        # 内容
        ax.text(box['x'] + box_w/2, box_y + box_h - 1.3, box['content'],
                ha='center', va='top', fontsize=10.5, linespacing=1.6, color='black', zorder=3)

    # 5. 绘制箭头
    arrow_style = dict(facecolor='#2C3E50', edgecolor='#2C3E50', shrinkA=0, shrinkB=0)
    
    def draw_arrow(x_start, x_end, label=None):
        ax.annotate('', xy=(x_end, box_y + box_h/2), xytext=(x_start + box_w, box_y + box_h/2),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#2C3E50'), zorder=2)
        if label:
            # 标签背景
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="#2C3E50", lw=0.5, alpha=0.9)
            ax.text(x_start + box_w + gap/2, box_y + box_h/2 + 0.2, label,
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2C3E50',
                    bbox=bbox_props, zorder=4)

    # 连接箭头
    draw_arrow(x_input, x_m1)
    draw_arrow(x_m1, x_m2, "Gap Definition")
    draw_arrow(x_m2, x_m3, "Forecast Dist.")
    draw_arrow(x_m3, x_out)

    plt.tight_layout()
    plt.savefig('Figure2_Framework.png', dpi=300, bbox_inches='tight')
    plt.show()

create_framework_diagram()