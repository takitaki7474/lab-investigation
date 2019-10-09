import numpy as np
import pandas as pd
import os
import time
import chainer
from chainer.links import VGG16Layers


def extract_feature(dataset, model, layer="fc7"):

    f_table = {}
    l_img_name = []
    l_feature = []
    l_label = []
    data_count = 0

    for data in dataset:
        tensor = data[0]
        label = data[1]
        img_name = "img_" + str(data_count)

        feature = model.extract([tensor], layers=[layer])[layer].data
        l_img_name.append(img_name)
        l_feature.append(feature)
        l_label.append(label)

        data_count += 1

        if data_count % 1000 == 0:
            print(data_count)

    f_table["img_name"] = l_img_name
    f_table["feature"] = l_feature
    f_table["label"] = l_label
    df = pd.DataFrame.from_dict(f_table)

    return df


def save_df(df, pkl="f_table.pkl"):

    if not os.path.exists("./feature"):
        os.mkdir("./feature")

    path = os.path.join("./feature", pkl)
    df.to_pickle(path)


if __name__ == "__main__":

    device = 0 #cpu#device=-1

    train, test = chainer.datasets.get_cifar10()
    model = VGG16Layers()

    if device >= 0:
    	import cupy as cp
    	import chainer.cuda

    	model.to_gpu()
    else:
    	cp = np

    start = time.time()
    df = extract_feature(train, model)
    print("抽出時間 {}秒".format(time.time()-start))

    save_df(df)
