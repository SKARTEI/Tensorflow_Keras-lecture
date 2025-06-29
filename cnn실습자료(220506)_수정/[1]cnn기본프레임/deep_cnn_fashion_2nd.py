#============================================
# 데이터 셋 준비

# 패션-MNIST 데이터셋 
from tensorflow.keras.datasets import fashion_mnist

# 패션-MNIST 데이터셋  불러오기(인터넷 통해 자동 다운로드)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
 
# 테스트용 속성 데이터
# x_test_encoded = x_test.reshape(10000, 784).astype('float32') / 255 
x_test_encoded = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255
#============================================


#===============================
# 저장했던 딥러닝 모델 불러오기
from tensorflow.keras.models import load_model  # 모델 불러오기 라이브러리
# model = load_model('my_model.h5') # 모델을 새로 불러옴
model = load_model('my_cnn_model.h5') # 모델을 새로 불러옴
#===============================

#===============================
# 속성(이미지)을 불러온 딥러닝 모델에 입력하여 예측값 획득
import numpy as np

index = 1 # 0~9999 사이에서 샘플 이미지 선택
x_input = x_test_encoded[index,:] # 테스트 셋에서 샘플 이미지 하나를 선택
#x_input = x_input.reshape(1, 784) # 1차원 배열(784)을 2차원 배열(1x784)의 형태로 맞추기
x_input = x_input.reshape(1, 28, 28, 1) 
# 1 : 1장의 이미지를 입력
# 28, 28 : 28x28 크기의 이미지를 의미
# 1 : 흑백 이미지를 의미   

prediction = model.predict(x_input) # 예측 수행
#softmax_sum = np.sum(prediction) # 소프트맥스 출력의 총 합은 1이어야 함
#===============================



# 클래스 이름
fashion_class = {
      0: 'T-shirt/top'
    , 1: 'Trouser'
    , 2: 'Pullover'
    , 3: 'Dress'
    , 4: 'Coat'
    , 5: 'Sandal'
    , 6: 'Shirt'
    , 7: 'Sneaker'
    , 8: 'Bag'
    , 9: 'Ankle boot'
}

#===============================
# 예측값이 맞는지 확인
class_predicted = np.argmax(prediction) # 클래스(예측)
class_actual = y_test[index]            # 클래스(정답)

class_predicted_name = fashion_class[class_predicted] # 클래스 이름(예측)
class_actual_name = fashion_class[class_actual] # 클래스 이름(정답)

print("클래스(예측) : %s" %class_predicted_name) # 클래스 이름(예측) 출력
print("클래스(정답) : %s" %class_actual_name) # 클래스 이름(정답) 출력


import matplotlib.pyplot as plt
plt.imshow(x_test[index], cmap=plt.cm.gray) # 샘플 이미지 표시
plt.show()
#=============================== 




