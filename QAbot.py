#%%
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imagehash


import time

def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
 
    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
 
    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations = 1)
 
    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations = 1)
 
    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations = 3)
 
    # 7. 存储中间图片 
    cv2.imwrite("binary.png", binary)
    cv2.imwrite("dilation.png", dilation)
    cv2.imwrite("erosion.png", erosion)
    cv2.imwrite("dilation2.png", dilation2)
 
    return dilation2
 
 
def findTextRegion(img):
    region = []
 
    # 1. 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt) 
 
        # 面积小的都筛选掉
        if(area < 1000):
            continue
 
        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
 
        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # print( "rect is: ")
        # print( rect)
 
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
 
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        if height < 40:
            maxh = max(box[0][1], box[2][1])
            idx = abs(box - maxh) < 5
            box[idx] += height
            minh = min(box[0][1], box[2][1])
            idx = abs(box - minh) < 5
            box[idx] -= height
            
        width = abs(box[0][0] - box[2][0])
        if height < 40:
            maxh = max(box[0][0], box[2][0])
            idx = abs(box - maxh) < 5
            box[idx] += width
            minh = min(box[0][0], box[2][0])
            idx = abs(box - minh) < 5
            box[idx] -= width
 
        # 筛选那些太细的矩形，留下扁的
        # if(height > width * 1.2):
        #     continue
 
        region.append(box)
 
    return region



#%%
from aip import AipOcr
config = {
    'appId': '18593767',
    'apiKey': 'XOLbgmM4rVOrItuq6w9odqZb',
    'secretKey': 'eGklNu6hVw7iDvQ29aALj7hN5VwHvLqh'
}
client = AipOcr(**config)
def get_file_content(file):
    with open(file, 'rb') as fp:
        return fp.read()
def img_to_str(image_path):
    image = get_file_content(image_path)
    result = client.basicGeneral(image)
    resultStr = ''
    try:
        for i in range(result['words_result_num']):
            resultStr += result['words_result'][i]['words']
            # resultStr += '\n'
        print(result)
    except:
        return ''
    return resultStr
# i = 22
# quesAns = img_to_str('{}/ans.png'.format(i)).split('\n')[:-1]
# print(quesAns)
# #%%
# i = 21
# img = cv2.imread('{}/screenshot.png'.format(i))
# cv2.imwrite("contours.png", img)
# image = get_file_content("contours.png")
# quesAns,result = img_to_str("contours.png")
# print(quesAns)

# # %%
# n = result['words_result_num']
# ith = 0
# for i in range(n):
#     tmp = result['words_result'][i]['words']
#     if (u'第' in tmp) & (u'题' in tmp) & (u'共' in tmp):
#         print(tmp)

        

# # %%
# '第9题·共1题'.split('·')


# %%
def crop_screecap(img):
    img_part = {}
    img_part['score'] = img[400:520,200:450]
    img_part['ques#'] = img[750:820,300:750]
    img_part['ques'] = img[820:1000,120:950]
    img_part['ans'] = img[1010:1920,220:840]

    return img_part

def save_crop(img_part, m):
    for key, value in img_part.items():
        cv2.imwrite('./' + str(m) + '/' + key + ".png", value)

def detect_ans(i):
    img = cv2.imread('{}/ans.png'.format(i))
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)
    lib = {}
    # 4. 用绿线画出这些找到的轮廓
    j = 1
    for box in region:
        minx, miny = min(box[:,1]), min(box[:,0])
        maxx, maxy = max(box[:,1]), max(box[:,0])
        sub_img = img[minx:maxx, miny:maxy]
        try:
            cv2.imwrite("{}/ans_{}.png".format(i,j), sub_img)
        except :
            return lib, j
        lib[j] = [(minx + maxx)/2, (miny + maxy)/2]
        j += 1
    return lib, j - 1

def check_lib(i):
    hashvalue = imagehash.phash(Image.open('{}/ques.png'.format(i)))
    text_quet = []
    correct_ans_idx = 0
    if hashvalue not in lib:
        ans, n_ans = detect_ans(i)
        text_quet = img_to_str('{}/ques.png'.format(i))
        text_quet = ''.join(text_quet)
        lib[hashvalue] = [i, text_quet, ans, n_ans, correct_ans_idx]
    return lib[hashvalue]


def search_ans(i):
    value = check_lib(i)
    if value[4] == 0:
        ans = []
        for j in range(1,value[3]+1):
            baiduai = ''
            try:
                baiduai = img_to_str("{}/ans_{}.png".format(value[0],j)).split('\n')[0]
            except:
                pass
            if baiduai == '':
                baiduai = 'adjfioiajdsiofja;jefo;eno;jjn;ioajd;fja'
            ans.append(baiduai)

        question = value[1]
        # choices = ['甲醛', '苯', '甲醇']
        choices = ans
        correct_ans_idx =  run_algorithm(2, question, choices) +1
        value[4] = correct_ans_idx
        value.append(ans)
    else:
        return value

    # hashvalue = hsh(img_part['ques'])
    
    # time.sleep(1)




import pickle

def save_dict(data, filename):
    with open(filename+'.p', 'wb') as fp:
        pickle.dump(lib, fp, protocol=pickle.HIGHEST_PROTOCOL)
def load_dict(filename):
    with open(filename + '.p', 'rb') as fp:
        data = pickle.load(fp)
        return data




def check_lib(i):
    hashvalue = imagehash.phash(Image.open('{}/ques.png'.format(i)))
    text_quet = []
    correct_ans_idx = 0
    if hashvalue not in lib:
        ans, n_ans = detect_ans(i)
        text_quet = img_to_str('{}/ques.png'.format(i))
        lib[hashvalue] = [i, text_quet, ans, n_ans, correct_ans_idx]
    return lib[hashvalue]






import requests
import webbrowser
import urllib.parse

# # 颜色兼容Win 10
from colorama import init,Fore
init()

def open_webbrowser(question):
    webbrowser.open('https://baidu.com/s?wd=' + urllib.parse.quote(question))

def open_webbrowser_count(question,choices):
    print('\n-- 方法2： 题目+选项搜索结果计数法 --\n')
    print('Question: ' + question)
    if '不是' in question:
        print('**请注意此题为否定题,选计数最少的**')

    counts = []
    for i in range(len(choices)):
        # 请求
        req = requests.get(url='http://www.baidu.com/s', params={'wd': question + choices[i]})
        content = req.text
        index = content.find('百度为您找到相关结果约') + 11
        content = content[index:]
        index = content.find('个')
        count = content[:index].replace(',', '')
        counts.append(count)
        #print(choices[i] + " : " + count)
    output(choices, counts)

def count_base(question,choices):
    print('\n-- 方法3： 题目搜索结果包含选项词频计数法 --\n')
    # 请求
    flag = 1
    while flag:
    # headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.80 Safari/537.36'}
        req = requests.get(url='http://www.baidu.com/s', params={'wd':question}, timeout=10)
        content = req.text
        if 'class="timeout-feedback hide' not in content:
            flag = False
    counts = []
    print('Question: '+question)
    if '不是' in question:
        print('**请注意此题为否定题,选计数最少的**')
    for i in range(len(choices)):
        counts.append(content.count(choices[i]))
        # print(choices[i] + " : " + str(counts[i]))
    return output(choices, counts)

def output(choices, counts):
    counts = list(map(int, counts))
    #print(choices, counts)

    # 计数最高
    index_max = counts.index(max(counts))

    # 计数最少
    index_min = counts.index(min(counts))
    

    if index_max == index_min:
        print(Fore.RED + "高低计数相等此方法失效！我猜1" + Fore.RESET)
        return 1

    for i in range(len(choices)):
        print()
        if i == index_max:
            # 绿色为计数最高的答案
            print(Fore.GREEN + "{0} : {1} ".format(choices[i], counts[i]) + Fore.RESET)
        elif i == index_min:
            # 红色为计数最低的答案
            print(Fore.MAGENTA + "{0} : {1}".format(choices[i], counts[i]) + Fore.RESET)
        else:
            print("{0} : {1}".format(choices[i], counts[i]))
    # print(index_max)
    return index_max

def run_algorithm(al_num, question, choices):
    if al_num == 0:
        open_webbrowser(question)
    elif al_num == 1:
        open_webbrowser_count(question, choices)
    elif al_num == 2:
        idx = count_base(question, choices)
    return idx
# %%
   

lib = {}
for i in range(1,23):

    img = cv2.imread('{}/screenshot.png'.format(i))
    img_part = crop_screecap(img)
    save_crop(img_part, i)
    search_ans(i)


# %%
i = 11
img = cv2.imread('{}/screenshot.png'.format(i))
img_part = crop_screecap(img)
save_crop(img_part, i)
search_ans(i)

# %%
