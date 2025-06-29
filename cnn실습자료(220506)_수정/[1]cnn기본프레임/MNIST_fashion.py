# 참고링크 : https://velog.io/@devseunggwan/Keras-Fashion-Mnist-%EC%8B%A4%EC%8A%B5
from tensorflow.keras.datasets import fashion_mnist

import matplotlib.pyplot as plt
import seaborn as sns

# MNIST 패션 데이터셋 불러오기(인터넷 통해 자동 다운로드)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# 데이터 셋 형태 확인
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# 정답 클래스(라벨) 별로 빈도수 세기
plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1) # 왼쪽에 그림 표시
sns.countplot(y_train, palette=['#' + ('{}'.format(i))*4 for i in range(10)])
plt.title('train labels count')

plt.subplot(1, 2, 2) # 오른쪽에 그림 표시
sns.countplot(y_test, palette=['#fb'+ ('{}'.format(i))*4 for i in range(10)])
plt.title('test labels count')

plt.plot()



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

# 학습 셋 이미지 확인(36개 만)
plt.figure(figsize=(12, 12))
for i in range(36):
    plt.subplot(6, 6, i+1) # i+1은 이미지가 삽입될 위치를 의미
    plt.suptitle('Train Images', fontsize=20)
    plt.title(fashion_class[y_train[i]])
    plt.imshow(x_train[i], cmap=plt.cm.gray)
    plt.axis("off")

plt.show()

# 테스트 셋 이미지 확인(36개 만)
plt.figure(figsize=(12, 12))
for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.suptitle('Test Images', fontsize=20)
    plt.title(fashion_class[y_test[i]])
    plt.imshow(x_test[i], cmap=plt.cm.gray)
    plt.axis("off")

plt.show()