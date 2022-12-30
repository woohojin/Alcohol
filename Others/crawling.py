# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 19:32:58 2022

@author: IncheonCUBE
"""

import time
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from selenium import webdriver  # pip install selenium
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import selenium
import os
import natsort
import urllib.request as req
selenium.__version__  # 3.141.0


# 크롬 드라이브를 통해 크롬 카카오맵 창 열기
path = 'USER CHROME DRIVER PATH'
source_url = 'https://smartstore.naver.com/wooridoga/category/aa273a12c1ff42a5973fab34b8a70aaa?st=POPULAR&free=false&dt=IMAGE&subscr=false&page=3&size=40'
options = webdriver.ChromeOptions()
options.add_argument("headless")
driver = webdriver.Chrome(path, options=options)  # 크롬 브라우저 실행
driver.get(source_url)

html = driver.page_source  # 현재 브라우저의 소스(html)
html

soup = BeautifulSoup(html, 'html.parser')
images = soup.find_all(name='img', attrs={'class': '_25CKxIKjAk'})
# 가져온 html 중에서 img태그 + class가 막걸리 사진들인 html구간을 images로 저장
images
page_urls = []

for image in images:
    page_url = image.get('src')
    page_urls.append(page_url)


# dir = "./openCV/img/" #디렉토리
# file_list = os.listdir(dir) #디렉토리 불러오기
# file_list_jpg = [file for file in file_list if file.endswith(".jpg")] #jpg파일만 가져오기
# if not file_list_jpg: #디렉토리에 jpg 파일이 비었을 경우 그냥 진행할경우 에러 발생을 처리
#     last_num = -1 #for문에서 +1을 해주어서 0부터 저장
# else:
#     order_list = natsort.natsorted(file_list_jpg) #오름차순으로 정렬
#     last_num = int(order_list[-1].replace(".jpg", "")) #last_num은 jpg파일 중 가장 마지막 숫자이름을 가져오고 그 이름에서 .jpg를 지우고 int형으로 바꾸어서 for문에 사용 할 수 있게 변경
#     last_num += 1 #마지막 위치를 기준으로 1을 더해 그 위치부터 파일을 저장
# print("Add %d.jpg | link: %s" %(last_num, link))


# =====================================================술마켓 크롤링======================================================

# enumerate(img_url) : index와 데이터(url링크)로 나눠서 조회
for index, link in enumerate(page_urls):
    # index : 순서 #link : 이미지의 url
    # req.urlretrieve : 인터넷에서 제공되는 내용을 파일로 저장
    # f'./img/{index}.jpg : 문자열 format형태, index별로 파일이 저장됨
    dir = "./openCV/img/"
    file_list = os.listdir(dir)
    file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
    if not file_list_jpg:
        last_num = 0
    else:
        order_list = natsort.natsorted(file_list_jpg)
        last_num = int(order_list[-1].replace(".jpg", ""))
        last_num += 1

    req.urlretrieve(link, f'./openCV/img/{last_num}.jpg')  # f"./img/0.jpg"
    print("Add %d.jpg | link: %s" % (last_num, link))

driver.close()

# =====================================================구글 크롤링 작은 이미지======================================================


SCROLL_PAUSE_SEC = 2
SCROLL_PAUSE_TIME = 2


def scroll_down():
    global driver
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:

        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);")

        time.sleep(SCROLL_PAUSE_SEC)

        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            try:
                driver.find_element_by_css_selector(
                    ".mye4qd").send_keys(Keys.ENTER)

            except:
                break

        last_height = new_height


keyword = input('검색할 태그를 입력하세요 : ')
url = 'https://www.google.com/search?q={}&tbm=isch&biw=1920&bih=960'.format(
    keyword)

path = 'USER CHROME DRIVER PATH'
options = webdriver.ChromeOptions()
options.add_argument("headless")
driver = webdriver.Chrome(path)
driver.get(url)

time.sleep(1)

scroll_down()

dir = './openCV/img/seonho/'
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
images = soup.find_all('img', attrs={'class': 'rg_i Q4LuWd'})

print('number of img tags: ', len(images))

n = 1
for i in images:

    try:
        imgUrl = i["src"]
    except:
        imgUrl = i["data-src"]

    with urllib.request.urlopen(imgUrl) as f:
        with open(dir + keyword + str(n) + '.jpg', 'wb') as h:
            img = f.read()
            h.write(img)

    n += 1

driver.close()


# =====================================================구글 크롤링 큰 이미지======================================================


keyword = input('검색할 태그를 입력하세요 : ')
url = 'https://www.google.com/search?q={}&tbm=isch&biw=1920&bih=960'.format(
    keyword)

path = 'USER CHROME DRIVER PATH'
options = webdriver.ChromeOptions()
options.add_argument("headless")
driver = webdriver.Chrome(path)
driver.get(url)

time.sleep(1)

scroll_down()

images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")

count = 1
for image in images:
    try:
        image.click()
        time.sleep(2)
        imgUrl = driver.find_element_by_xpath(
            'html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute("src")
        urllib.request.urlretrieve(imgUrl, str(count) + ".jpg")
        count = count + 1
    except:
        pass
driver.close()
