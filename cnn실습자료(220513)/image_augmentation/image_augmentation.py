
from tensorflow.keras.preprocessing.image import ImageDataGenerator


augmentation = ImageDataGenerator(
                                  #zoom_range=0.3, # 확대/축소 랜덤 적용
                                  width_shift_range=0.2, # 좌우 시프트 랜덤 적용
                                  height_shift_range=0.2, # 상하 시프트 랜덤 적용
                                  horizontal_flip=True, # 좌우 반전 랜덤 적용
                                  vertical_flip=True, # 상하 반전 랜덤 적용
                                  rotation_range=90 # 랜덤 회전 적용
                                  ) 

i = 0
for batch in augmentation.flow_from_directory('', classes=['class_00'], # ▶▶원본 이미지 저장 폴더
                               batch_size=1,
                               save_to_dir='class_00_augmentation', # ▶▶변형/조작된 이미지 저장 폴더
                               save_prefix='img', # 저장할 이미지 파일명 앞 부분
                               save_format='jpg'): # 저장할 이미지 포맷
    i += 1
    if i >= 100:
        break  # 이미지 100장을 생성하고 마칩니다(생성 숫자 조정 가능)

