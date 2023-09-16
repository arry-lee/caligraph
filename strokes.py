import io
import os.path
import random
import urllib.parse

import numpy as np
import requests
from PIL import Image

# 引入本地缓存
from strokes_code import STOKES_CODE


def char_frames(word, cache='./cache',vibe=True):
    """
    将一个汉字变成书写动画帧
    """
    word = hex((ord(word)))[2:]
    word = urllib.parse.quote(word)
    if not os.path.exists(cache):
        os.makedirs(cache)
    img_path = os.path.join(cache, f"{word}.png")
    url = f"http://www.hanzi5.com/assets/bishun/stroke/{word}-fenbu.png"
    # gif = f"http://www.hanzi5.com/assets/bishun/stroke/{word}-bishun.gif"
    try:
        img = Image.open(img_path)
    except:
        img = Image.open(io.BytesIO(requests.get(url).content))
        img.save(img_path)
    
    img = img.convert('L')
    w = 150
    x = 0
    y = 0
    imgs = []
    while True:
        imglst = img.crop((x, y, x + w, y + w))
        imgs.append(imglst)
        x = x + w + 3
        if x > img.width:
            x = 0
            y = y + w + 3
        if y > img.height:
            break
    imgs.pop()
    while np.all(np.array(imgs[-1]) == 255):
        imgs.pop()
    BLUE = 167
    GRAY = 237
    RED = 135
    # BLACK = 48
    # WHITE = 255
    out = [Image.new('L',(w,w),255)]#
    # base = Image.new('L',(w,w),255)
    border = imgs[0].point(lambda x: x if x == BLUE else 255)
    for index, i in enumerate(imgs):
        im = out[-1].copy()
        msk = i.point(lambda x: 0 if x == RED else 255)
        if vibe:
            msk = msk.rotate(random.randint(-3,3),fillcolor=255,translate=(random.randint(-1,1),random.randint(-1,1)))

        im.paste(msk,mask=msk.point(lambda x:255-x))
        # im.paste(border,mask=border.point(lambda x: 0 if x==255 else 255))
        # im.save(f'{index}.jpg')
        out.append(msk)#x == BLUE or fixme

    output_file = f"{word}.gif"
    duration = 50  # 帧之间的延迟时间（以毫秒为单位）
    loop = 0  # 动画循环次数（0表示无限循环）

    # out[0] = imgs[0].point(lambda x: x if x == BLUE else 255)
    # 使用第一个图像创建一个新的GIF对象，将其他图像追加为帧
    # out[0].save(output_file, save_all=True, append_images=out[1:], duration=duration, loop=loop)
    return out


def char_strokes(word):
    return STOKES_CODE[word]
