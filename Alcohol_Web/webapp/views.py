from django.shortcuts import render
from .models import Alcohol
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2


def image(request):
    return render(request, "image.html")


def camera(request):
    return render(request, "camera.html", {"bool": bool})


def predict(request):
    if request.method != "POST":
        return render(request, "image.html")
    else:
        # predict하기 위한 img_path
        # file/naru1.jpg
        img_path = "file/" + request.POST["picture"]

        img_height = 296  # 인식된 이미지 사이즈 조정
        img_width = 250  # 인식된 이미지 사이즈 조정

        # 이미지를 불러오고 predict 할 수 있도록 배열로 변경
        predict_img = keras.preprocessing.image.load_img(
            img_path, target_size=(img_height, img_width))
        predict_array = keras.preprocessing.image.img_to_array(predict_img)
        predict_array = tf.expand_dims(predict_array, 0)

        model = tf.keras.models.load_model("model/alcohol_model.h5")

        predictions = model.predict(predict_array)
        score = tf.nn.softmax(predictions[0])

        class_names = ["geumjung", "ihwa", "naru", "peow", "red", "seonho"]

        predict_alcohol = class_names[np.argmax(score)]
        predict_score = 100 * np.max(score)

        # DB를 이용하려고 했지만 sqlite의 경우 한글이 깨지는 현상이 발생해서 model을 이용해서 생성
        # 클라우드 서비스를 이용하여 업로드 후 사용가능 함
        if predict_alcohol == "naru":
            alcohol_info = Alcohol(name="나루 생 막걸리",
                                   abv=6,
                                   price=7000,
                                   standard="935ml",
                                   material="정제수, 쌀(국내산), 국, 효모(밀함유)",
                                   company="한강주조")
            alcohol_info.save()
        elif predict_alcohol == "geumjung":
            alcohol_info = Alcohol(name="금정산성 막걸리",
                                   abv=8,
                                   price=3100,
                                   standard="750ml",
                                   material="백미, 밀(누룩), 정제수",
                                   company="금정산성 토산주")
            alcohol_info.save()
        elif predict_alcohol == "ihwa":
            alcohol_info = Alcohol(name="이화백주",
                                   abv=6,
                                   price=12000,
                                   standard="940ml",
                                   material="정제수, 쌀(국내산), 누룩, 설탕, 물엿, 밀, 아스파탐",
                                   company="이화백주")
            alcohol_info.save()
        elif predict_alcohol == "red":
            alcohol_info = Alcohol(name="붉은 원숭이 막걸리",
                                   abv=10.8,
                                   price=8500,
                                   standard="375ml",
                                   material="쌀(국내산 경기미 100%), 누룩, 홍국, 정제수",
                                   company="술샘")
            alcohol_info.save()
        elif predict_alcohol == "seonho":
            alcohol_info = Alcohol(name="선호 생 막걸리",
                                   abv=6,
                                   price=2100,
                                   standard="750ml",
                                   material="쌀(국내산), 누룩, 효모, 정제효소, 천연감미료, 정제수",
                                   company="김포금쌀탁주영농조합법인")
            alcohol_info.save()
        elif predict_alcohol == "song":
            alcohol_info = Alcohol(name="송명섭 막걸리",
                                   abv=6,
                                   price=2000,
                                   standard="500ml",
                                   material="쌀(국내산), 곡자, 정제수",
                                   company="태인합동주조장")
            alcohol_info.save()

        # .2f = 소수점 2자리까지 표시
        result = "{:.2f}%의 확률로 {} 입니다.".format(
            predict_score, alcohol_info.name)

        # predict를 한 이후에 image.html에서 이미지가 보이지 않는 문제를 위해 추가된 img_path
        # /file/naru.jpg
        img_path = "/" + img_path

        # img가 있는지 확인
        if img_path == "":
            bool = 0
        else:
            bool = 1

        return render(request, "image.html", {"result": result, "info": alcohol_info, "pic": img_path, "bool": bool})


def predict_camera(request):
    cap = cv2.VideoCapture(0)
    # width = 640, height = 480
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while(True):
        ret, img_color = cap.read()

        if ret == False:
            break

        cv2.imshow('bgr', img_color)

        key = cv2.waitKey(1)

        if key == 32:
            img_color = cv2.resize(
                img_color, (250, 296), interpolation=cv2.INTER_AREA)
            cv2.imshow('img_color', img_color)
            cv2.waitKey(0)
            cap.release()
            cv2.destroyAllWindows()

            img_height = 296  # 인식된 이미지 사이즈 조정
            img_width = 250  # 인식된 이미지 사이즈 조정

            # 이미지를 불러오고 predict 할 수 있도록 배열로 변경
            predict_array = keras.preprocessing.image.img_to_array(img_color)
            predict_array = tf.expand_dims(predict_array, 0)

            model = tf.keras.models.load_model("model/first_model.h5")

            predictions = model.predict(predict_array)
            score = tf.nn.softmax(predictions[0])

            class_names = ["geumjung", "ihwa", "naru", "peow", "red", "seonho"]

            predict_alcohol = class_names[np.argmax(score)]
            predict_score = 100 * np.max(score)
            print("predict_score ==================", predict_score)
            # DB를 이용하려고 했지만 sqlite의 경우 한글이 깨지는 현상이 발생해서 model을 이용해서 생성
            # 클라우드 서비스를 이용하여 업로드 후 사용가능 함
            if predict_alcohol == "naru":
                alcohol_info = Alcohol(name="나루 생 막걸리",
                                       abv=6,
                                       price=7000,
                                       standard="935ml",
                                       material="정제수, 쌀(국내산), 국, 효모(밀함유)",
                                       company="한강주조")
                alcohol_info.save()
            elif predict_alcohol == "geumjung":
                alcohol_info = Alcohol(name="금정산성 막걸리",
                                       abv=8,
                                       price=7000,
                                       standard="750ml",
                                       material="백미, 밀(누룩), 정제수",
                                       company="금정산성 토산주")
                alcohol_info.save()
            elif predict_alcohol == "ihwa":
                alcohol_info = Alcohol(name="이화백주",
                                       abv=6,
                                       price=7000,
                                       standard="940ml",
                                       material="정제수, 쌀(국내산), 누룩, 설탕, 물엿, 밀, 아스파탐",
                                       company="이화백주")
                alcohol_info.save()
            elif predict_alcohol == "red":
                alcohol_info = Alcohol(name="붉은 원숭이 막걸리",
                                       abv=10.8,
                                       price=7000,
                                       standard="375ml",
                                       material="쌀(국내산 경기미 100%), 누룩, 홍국, 정제수",
                                       company="술샘")
                alcohol_info.save()
            elif predict_alcohol == "seonho":
                alcohol_info = Alcohol(name="선호 생 막걸리",
                                       abv=6,
                                       price=7000,
                                       standard="750ml",
                                       material="쌀(국내산), 누룩, 효모, 정제효소, 천연감미료, 정제수",
                                       company="김포금쌀탁주영농조합법인")
                alcohol_info.save()
            elif predict_alcohol == "peow":
                alcohol_info = Alcohol(name="표문 막걸리",
                                       abv=6,
                                       price=7000,
                                       standard="500ml",
                                       material="쌀(국내산), 곡자, 정제수",
                                       company="태인합동주조장")
                alcohol_info.save()

            # .2f = 소수점 2자리까지 표시
            result = "{:.2f}%의 확률로 {} 입니다.".format(
                predict_score, alcohol_info.name)
            bool = 1
            return render(request, "camera.html", {"result": result, "info": alcohol_info, "bool": bool})


def handle_upload(f):
    with open("file/" + f.name, "wb+") as file:
        for ch in f.chunks():
            file.write(ch)


def picture(request):
    if request.method != "POST":
        return render(request, "pictureform.html")
    else:
        fname = request.FILES["picture"].name
        handle_upload(request.FILES["picture"])
        return render(request, "picture.html", {"fname": fname})
