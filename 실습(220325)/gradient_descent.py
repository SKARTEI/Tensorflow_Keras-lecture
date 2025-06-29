w=10 # 가중치 초깃값
learning_rate = 0.2 # 학습률
max_iterations = 10 # 갱신 반복 횟수

# 오차 함수를 람다식으로 정의
error_func = lambda w: (w-3)**2 + 10

# 오차 함수의 미분
gradient = lambda w: 2*w-6

# 가중치 초깃값에 대한 오차미분과 오차함수 값
print("가중치: %f,   오차미분: %f   오차함수: %f" %(w, gradient(w), error_func(w)))

# 경사 하강법 실행
for i in range(max_iterations):
    w = w - learning_rate * gradient(w)
    # 가중치 갱신 뒤 대한 오차미분과 오차함수 값
    print("가중치: %f,   오차미분: %f   오차함수: %f" %(w, gradient(w), error_func(w))) 