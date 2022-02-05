import random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from matplotlib.font_manager import FontProperties

 
def draw_heatmap(data, filename):

 
    #作图阶段
    fig = plt.figure()
    #定义画布为1*1个划分，并在第1个位置上进行作图
    ax = fig.add_subplot(111)
    #作图并选择热图的颜色填充风格，这里选择hot
    im = ax.imshow(data, cmap=plt.cm.hot_r)
    #增加右侧的颜色刻度条
    plt.colorbar(im)
    #增加标题
    plt.title("weight heatmap")
    #show
    plt.show()
    plt.savefig(filename)
 
# d = draw()