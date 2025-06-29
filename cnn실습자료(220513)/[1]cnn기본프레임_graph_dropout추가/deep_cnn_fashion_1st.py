
#============================================
# 딥러닝 모델 생성 관련 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# 패션-MNIST 데이터셋  
from tensorflow.keras.datasets import fashion_mnist

# 정답 데이터 변환(전처리) 관련
from tensorflow.keras import utils
#============================================

# 패션-MNIST 데이터셋 불러오기(인터넷 통해 자동 다운로드)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 학습용 속성 데이터
# x_train_encoded = x_train.reshape(60000, 784).astype('float32') / 255 
x_train_encoded = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255

# 테스트용 속성 데이터
# x_test_encoded = x_test.reshape(10000, 784).astype('float32') / 255 
x_test_encoded = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255

# 정답 클래스에 원 핫 인코딩 적용(해당 클래스 자리만 1, 나머진 0)
y_train_encoded = utils.to_categorical(y_train, num_classes=10)
y_test_encoded = utils.to_categorical(y_test, num_classes=10)

# 딥러닝 모델의 구조 설정
# model = Sequential()
# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# 컨볼루션 층(convolution layer) - 풀링 포함
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

# 일반적인 신경망 층(fully connected layer)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
# model.add(Dense(200, activation='relu')) # 필요시 자유롭게 은닉층 추가 가능
model.add(Dense(10, activation='softmax'))

# 딥러닝 학습 설정
# (loss : 오차함수 종류, optimizer : 최적화 알고리즘, metrics : 평가 기준)
model.compile(optimizer='adam', # sgd or 'adam'
              loss='mean_squared_error', # 'mean_squared_error' or 'categorical_crossentropy'
              metrics=['acc'])

# 딥러닝 학습 실행
model.fit(x_train_encoded, y_train_encoded, epochs=10, batch_size=100)

# 딥러닝 모델 학습 후 성능 측정(정확도)
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test_encoded, y_test_encoded)[1]))

# 딥러닝 모델 저장
# model.save('my_model.h5')   # 모델을 컴퓨터에 저장
model.save('my_cnn_model.h5')   # 모델을 컴퓨터에 저장


