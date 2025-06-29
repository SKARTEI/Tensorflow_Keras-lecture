import numpy as np
import glob
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 클래스 정의
img_00 = ".\dataset\class_00" # 클래스 0번 (가위)
img_01 = ".\dataset\class_01" # 클래스 1번 (바위)
img_02 = ".\dataset\class_02" # 클래스 2번 (보)
img_03 = ".\dataset\class_03" # 클래스 3번 (손날)

ver = 50 # (이미지 리사이즈 후) 세로 픽셀수
hor = 100 # (이미지 리사이즈 후) 가로 픽셀수
X_all = [] # 속성 데이터가 들어갈 변수 생성
Y_all = [] # 정답 클래스가 들어갈 변수 생성

#0000000000000000000000000000000000
# 클래스 0번 속성 생성
files_00 = sorted(glob.glob(img_00 + "\*.jpg"))
num_00 = len(files_00)
X = [] # 비어있는 배열 생성

for i, filepath in enumerate(files_00):
    img = image.load_img(filepath,                # 이미지의 실제 경로
                         color_mode='grayscale',  # 이미지를 무조건 흑백으로 불러오기
                         target_size = (ver, hor)) # 세로50픽셀x가로100픽셀 사이즈로 이미지 불러오기
    img_array = image.img_to_array(img) # 이미지를 배열 형식으로 변환
    X.append(img_array) # 이미지를 하나씩 추가하여 한 클래스의 이미지 셋(집합) 생성
    
X_00 = np.array(X) # 한 클래스의 이미지 셋을 배열 형태로 변환

# 클래스 0번 정답 클래스 생성
Y_00 = 0 * np.ones(num_00)
#0000000000000000000000000000000000



#1111111111111111111111111111111111
# 클래스 1번 속성 생성
files_01 = sorted(glob.glob(img_01 + "\*.jpg"))
num_01 = len(files_01)
X = [] # 비어있는 배열 생성

for i, filepath in enumerate(files_01):
    img = image.load_img(filepath,                # 이미지의 실제 경로
                         color_mode='grayscale',  # 이미지를 무조건 흑백으로 불러오기
                         target_size = (ver, hor)) # 세로50픽셀x가로100픽셀 사이즈로 이미지 불러오기
    img_array = image.img_to_array(img) # 이미지를 배열 형식으로 변환
    X.append(img_array) # 이미지를 하나씩 추가하여 한 클래스의 이미지 셋(집합) 생성
    
X_01 = np.array(X) # 한 클래스의 이미지 셋을 배열 형태로 변환

# 클래스 1번 정답 클래스 생성
Y_01 = 1 * np.ones(num_01)
#1111111111111111111111111111111111



#2222222222222222222222222222222222
# 클래스 2번 속성 생성
files_02 = sorted(glob.glob(img_02 + "\*.jpg"))
num_02 = len(files_02)
X = [] # 비어있는 배열 생성

for i, filepath in enumerate(files_02):
    img = image.load_img(filepath,                # 이미지의 실제 경로
                         color_mode='grayscale',  # 이미지를 무조건 흑백으로 불러오기
                         target_size = (ver, hor)) # 세로50픽셀x가로100픽셀 사이즈로 이미지 불러오기
    img_array = image.img_to_array(img) # 이미지를 배열 형식으로 변환
    X.append(img_array) # 이미지를 하나씩 추가하여 한 클래스의 이미지 셋(집합) 생성
    
X_02 = np.array(X) # 한 클래스의 이미지 셋을 배열 형태로 변환

# 클래스 2번 정답 클래스 생성
Y_02 = 2 * np.ones(num_02)
#2222222222222222222222222222222222



#3333333333333333333333333333333333
# 클래스 3번 속성 생성
files_03 = sorted(glob.glob(img_03 + "\*.jpg"))
num_03 = len(files_03)
X = [] # 비어있는 배열 생성

for i, filepath in enumerate(files_03):
    img = image.load_img(filepath,                # 이미지의 실제 경로
                         color_mode='grayscale',  # 이미지를 무조건 흑백으로 불러오기
                         target_size = (ver, hor)) # 세로50픽셀x가로100픽셀 사이즈로 이미지 불러오기
    img_array = image.img_to_array(img) # 이미지를 배열 형식으로 변환
    X.append(img_array) # 이미지를 하나씩 추가하여 한 클래스의 이미지 셋(집합) 생성
    
X_03 = np.array(X) # 한 클래스의 이미지 셋을 배열 형태로 변환

# 클래스 3번 정답 클래스 생성
Y_03 = 3 * np.ones(num_03)
#3333333333333333333333333333333333

# 속성 및 정답 클래스 합치기
X_all = np.concatenate((X_00, X_01, X_02, X_03), axis=0) # 모든 클래스 속성 합치기
Y_all = np.concatenate((Y_00, Y_01, Y_02, Y_03), axis=0) # 모든 클래스 정답 합치기

# 학습셋과 테스트셋의 구분
from sklearn.model_selection import train_test_split

# 학습셋과 테스트셋의 구분
x_train, x_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=0.3, random_state=0)

xy = (x_train, x_test), (y_train, y_test)
np.save(".\dataset_hand.npy", xy)







