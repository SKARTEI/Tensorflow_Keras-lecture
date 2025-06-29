# 딥러닝 모델 생성 관련
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 속싱 및 정답 데이터 변환(전처리) 관련
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils
import numpy as np

# 데이터 셋 불러오기 관련
import pandas as pd 


# 데이터 셋 불러오기
df = pd.read_csv('./dataset/data_01.txt', 
                 header=None, 
                 names = ["Class", "Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"])

# 데이터 분리(속성과 정답클래스)
dataset = df.values
X = dataset[:,1:5] # 속성
X_np = np.array(X, dtype=np.float) # 케라스 딥러닝 모델에 맞도록 변수 형식만 변경

Y_obj = dataset[:,0] # 정답 클래스
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj) # 정답 클래스 문자열->숫자 변환
Y_encoded = utils.to_categorical(Y) # 원 핫 인코딩 적용

# 딥러닝 모델의 구조 설정
model = Sequential() # 딥러닝 모델 선언
model.add(Dense(16, input_dim=4, activation='relu')) # 입력층 + 은닉층1
model.add(Dense(16, activation='relu')) # 은닉층2
model.add(Dense(32, activation='relu')) # 은닉층3
model.add(Dense(3, activation='softmax')) # 출력층

# 딥러닝 학습 설정
# (loss : 오차함수 종류, optimizer : 최적화 알고리즘, metrics : 평가 기준)
model.compile(loss='mean_squared_error',
            optimizer='sgd',
            metrics=['acc'])

# 딥러닝 모델 학습 실행
model.fit(X_np, Y_encoded, epochs=10, batch_size=15)

# 딥러닝 모델 학습 결과(정확도)
print("\n Accuracy: %.4f" % (model.evaluate(X_np, Y_encoded)[1]))










