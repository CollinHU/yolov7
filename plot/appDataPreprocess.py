import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import io

def processJson(data):
    resultX = []
    resultY = []
    for stroke in data:
        Xs = []
        Ys = []
        for key in stroke.keys():
            for coor in stroke[key]:
                Xs.append(coor['x'])
                Ys.append(coor['y'])
        if len(Xs) != 0:
            resultX.append(np.array(Xs))
            resultY.append(np.array(Ys))
    return resultX, resultY

def plotImg(resultX_scaled, resultY_scaled, 
    line_width, save_fig = True, imgPath = 'test.jpg'):
    min_scaled_X = min([ele.min() for ele in resultX_scaled])
    max_scaled_X = max([ele.max() for ele in resultX_scaled])
    min_scaled_Y = min([ele.min() for ele in resultY_scaled])
    max_scaled_Y = max([ele.max() for ele in resultY_scaled])
    
    dpi = 72
    width_height_ratio =  (max_scaled_X - min_scaled_X) / (max_scaled_Y - min_scaled_Y)
    #print(width_height_ratio)
    imgW = 775/dpi * width_height_ratio
    imgH = 796/dpi
    
    fig, ax = plt.subplots(figsize=(imgW, imgH), dpi=dpi)

    for x, y in zip(resultX_scaled, resultY_scaled):
        ax.plot(x, (1 + max_scaled_Y - y), '-', color='black', linewidth=line_width)
    plt.axis('off')
    ax.set_xlim(min_scaled_X - 1,max_scaled_X + 1)
    ax.set_ylim(0, 10)
    
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    data = []

    if save_fig:
        plt.savefig(imgPath, dpi = dpi, bbox_inches=extent)
    else:
        io_buf = io.BytesIO()

        fig.savefig(io_buf,format = 'raw', dpi=dpi, bbox_inches= extent)
        io_buf.seek(0)
        bbox = ax.get_window_extent().bounds
        data = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                     newshape=(int(bbox[3]), int(bbox[2]), -1))[:,:,:3]
        io_buf.close()

    plt.close(fig)
    
    scaled_width = max_scaled_X - min_scaled_X
    scaled_height = max_scaled_Y - min_scaled_Y
    return scaled_width, scaled_height, data

def rescaleCoordinate(resultX, resultY):
    minX = min([ele.min() for ele in resultX])
    minY = min([ele.min() for ele in resultY])
    
    maxX = max([ele.max() for ele in resultX])
    maxY = max([ele.max() for ele in resultY])
    
    height = maxY - minY
    width = maxX - minX
    left_up_corner= (minX, minY)
    
    resultY_scaled = []
    for arr in resultY:
        resultY_scaled.append(8 * (arr - minY) / height + 1)
    resultX_scaled = []
    for arr in resultX:
        resultX_scaled.append(8 * (arr - minX) / height + 1)
        
    return left_up_corner, width, height, resultX_scaled, resultY_scaled