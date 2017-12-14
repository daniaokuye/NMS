import numpy as np
import matplotlib.pyplot as plt
import random
import os
import re
from PIL import Image, ImageDraw

visual = {1: 'person', 2: 'bird', 3: 'cat', 4: 'cow', 5: 'dog', 6: 'horse',
          7: 'sheep', 8: 'aeroplane', 9: 'bicycle', 10: 'boat', 11: 'bus', 12: 'car',
          13: 'motorbike', 14: 'train', 15: 'bottle', 16: 'chair', 17: 'diningtable',
          18: 'pottedplant', 19: 'sofa', 20: 'tvmonitor'}


def read_data(filepath):
    """

    :param filepath: input
    :return: a dict for
    """
    path_dir = os.listdir(filepath)

    file = {}
    for dir in path_dir:
        if 'NMS' in dir:
            continue
        x = int(re.findall('\d+', dir)[0])
        if file.get(x):
            file[x].append(dir)
        else:
            file[x] = [dir]
    for key, con in file.items():
        rearray = []
        order = ['image', 'pred', 'gt']
        for i in order:
            for j in con:
                if i in j:
                    rearray.append(j)
                    break

        file[key] = rearray
        for i in range(len(file[key])):
            file[key][i] = filepath + file[key][i]

    return file


def show(data, boxes):
    plt.imshow(data)
    ax = plt.gca()
    color = plt.cm.hsv(np.linspace(0, 1, 20)).tolist()
    h, w, c = data.shape
    for box in boxes:
        xmin, xmax, ymin, ymax = coor_point(box)
        *_, label = box
        h_box = ymax - ymin
        w_box = xmax - xmin

        # just for show
        if xmin < 0:
            xmin = 2 / w
        if ymin < 0:
            ymin = 2 / w
        if w_box > 1:
            w_box = (w - 2) / w
        if h_box > 1:
            h_box = (h - 2) / h
        coor = (int(xmin * w), int(ymin * h)), int(w_box * w), int(h_box * h)
        ax.add_patch(
            plt.Rectangle(*coor, fill=False, edgecolor=color[int(label)], linewidth=2))  # random.randint(0, 20)
        # plt.show()


def show_PIL(data, boxes, dicts, color, label=1, scale=1.8):
    data = Image.fromarray(data)
    w, h = data.size
    data = data.resize((int(w * scale), int(h * scale)), Image.ANTIALIAS)
    w, h = data.size
    draw = ImageDraw.Draw(data)
    for box in boxes:
        if len(box) > 4:
            xmin, xmax, ymin, ymax = coor_point(box)
            *_, label = box
        else:
            xmin, xmax, ymin, ymax = coor_point(box)
            label = label
            color = (0, 200, 200)
        # just for show
        if xmin < 0:
            xmin = 2 / w
        if ymin < 0:
            ymin = 2 / w
        if xmax > 1:
            xmax = (w - 2) / w
        if ymax > 1:
            ymax = (h - 2) / h
        coor = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))
        draw.rectangle(coor, outline=color)
        draw.text(coor[0:2], dicts[int(label)], fill=color)
    data.show()


def arr_img(data):
    """
    arrange datas in a row
    :param data:
    :return:
    """
    if len(data.shape) == 4:
        batch, w, h, c = data.shape
        pad = np.ones((c, 10, w))
        img = data[0]
        for i in range(1, batch):
            np.hstack((img, pad))
            np.hstack((img, data[i]))
    else:
        img=data

    return img


def label_i_boxes(boxes, min_threshold=0.1):
    """
    for each labels in boxes,apply Non-maxxing supress
    :param boxes:
    :param labels:numbers of label in these boxes
    :param min_threshold:
    :return:out,information of boxes for every label
    """
    out = []
    row, col = boxes.shape
    labels = set(boxes[:, -1])
    for i in labels:
        label_I = np.where(boxes[:, -1] == i)  # index of boxes when label is i
        boxex_I = boxes[label_I]
        newbox_I = nms_boxes(boxex_I, min_threshold)
        if type(newbox_I) != int:
            if type(newbox_I[0]) == int:
                out.append(newbox_I.tolist())
            else:
                out += newbox_I.tolist()
    return out


def nms_boxes(boxes, min_threshold):
    """
    delete if boxes has meet NMS scheme
    :param boxes:predict boxes
    :param min_threshold:min threshold of confidence,le.0.1
    :return:
    """
    has_reffered = []
    while True:
        if not len(boxes):
            return boxes
        max_conf = np.max(boxes[:, -2])
        if max_conf < min_threshold:
            return -1
        conf = np.where(boxes[:, -2] == max_conf)[0]  # index
        max_conf = conf[0]
        ref = boxes[max_conf, :]  # corresponding box info
        remove = []
        done = []
        for i in range(len(boxes)):
            if i == max_conf:
                continue
            lap = overlap_(boxes[i], ref)
            if lap > 0.45:
                remove.append(i)
            else:
                done.append(max_conf)
        done = list(set(done))
        remove += done
        # all is done but those don't overlap
        boxes = np.delete(boxes, remove, 0)  # axis=0 delet line

        if len(done):
            has_reffered.append(ref.tolist())
        else:
            break
    if len(has_reffered):
        boxes = np.vstack((boxes, has_reffered))
    return boxes


def overlap_(sujects, objects):
    """
    cal the jaccard overlap between those two
    :param sujects:
    :param objects:
    :return: overlap for delete
    """
    esp = 1E-15
    minx_sj, maxx_sj, miny_sj, maxy_sj = coor_point(sujects)
    minx_oj, maxx_oj, miny_oj, maxy_oj = coor_point(objects)
    minx = minx_oj if minx_oj > minx_sj else minx_sj
    miny = miny_oj if miny_oj > miny_sj else miny_sj
    maxx = maxx_oj if maxx_oj < maxx_sj else maxx_sj
    maxy = maxy_oj if maxy_oj < maxy_sj else maxy_sj
    w_sj, h_sj = maxx_sj - minx_sj, maxy_sj - miny_sj
    w_oj, h_oj = maxx_oj - minx_oj, maxy_oj - miny_oj
    w_ol, h_ol = maxx - minx, maxy - miny
    sj = w_sj * h_sj
    oj = w_oj * h_oj
    ol = w_ol * h_ol  # overlap
    if ol < 0:
        ol = 0
    return ol / (sj + oj - ol + esp)


def coor_point(objects):
    cx, cy, w, h = objects[0:4]
    # minx,maxx,miny,maxy
    coor = [cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2]
    for i in range(len(coor)):
        if coor[i] > 1:
            coor[i] = 1.0
        if coor[i] < 0.0:
            coor[i] = 0
    return coor


if __name__ == "__main__":
    filepath = '/home/flag54/Documents/NMS/litti/'
    file = read_data(filepath)
    # read pyb
    for items in file.values():
        if len(items) == 3:
            im, box, gt = items
        if len(items) == 2:
            im, box = items
        img = np.load(im).astype('uint8')
        img = arr_img(img)

        boxes = np.load(box)
        # gts = np.load(gt)

        lb = label_i_boxes(boxes)  # 21 classes
        fp = box.split('_')[0] + '_NMS.npy'
        np.save(fp, np.array(lb))
        # show(img, lb)
        # lb += gts.tolist()
        show_PIL(img, lb, visual, (255, 110, 0))
        print('o')
