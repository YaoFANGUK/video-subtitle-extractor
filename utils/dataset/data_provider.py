# encoding:utf-8
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.dataset.data_util import GeneratorEnqueuer

DATA_FOLDER = "data/dataset/mlt/"


def get_training_data():
    img_files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(os.path.join(DATA_FOLDER, "image")):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    img_files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(img_files)))
    return img_files


def load_annoataion(p):
    bbox = []
    with open(p, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")
        x_min, y_min, x_max, y_max = map(int, line)
        bbox.append([x_min, y_min, x_max, y_max, 1])
    return bbox


def generator(vis=False):
    image_list = np.array(get_training_data())
    print('{} training images in {}'.format(image_list.shape[0], DATA_FOLDER))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        for i in index:
            try:
                im_fn = image_list[i]
                im = cv2.imread(im_fn)
                h, w, c = im.shape
                im_info = np.array([h, w, c]).reshape([1, 3])

                _, fn = os.path.split(im_fn)
                fn, _ = os.path.splitext(fn)
                txt_fn = os.path.join(DATA_FOLDER, "label", fn + '.txt')
                if not os.path.exists(txt_fn):
                    print("Ground truth for image {} not exist!".format(im_fn))
                    continue
                bbox = load_annoataion(txt_fn)
                if len(bbox) == 0:
                    print("Ground truth for image {} empty!".format(im_fn))
                    continue

                if vis:
                    for p in bbox:
                        cv2.rectangle(im, (p[0], p[1]), (p[2], p[3]), color=(0, 0, 255), thickness=1)
                    fig, axs = plt.subplots(1, 1, figsize=(30, 30))
                    axs.imshow(im[:, :, ::-1])
                    axs.set_xticks([])
                    axs.set_yticks([])
                    plt.tight_layout()
                    plt.show()
                    plt.close()
                yield [im], bbox, im_info

            except Exception as e:
                print(e)
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    gen = get_batch(num_workers=2, vis=True)
    while True:
        image, bbox, im_info = next(gen)
        print('done')
