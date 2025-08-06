import tensorflow as tf
import os
import json
from Sounddetect import get_commands

def convert_model():
    # 기존 모델 로드
    old_model_path = 'speech_recognition_model.h5'
    if not os.path.exists(old_model_path):
        print(f"오류: 기존 모델 파일 '{old_model_path}'를 찾을 수 없습니다.")
        return
        
    try:
        # 모델 로드
        model = tf.keras.models.load_model(old_model_path)
        print("기존 모델을 성공적으로 로드했습니다.")
        
        # 새 형식으로 저장
        new_model_path = 'speech_recognition_model.keras'
        model.save(new_model_path)
        print(f"모델을 새 형식('{new_model_path}')으로 저장했습니다.")
        
        # 모델 정보 저장
        commands = get_commands()
        model_info = {
            'commands': commands,
            'input_shape': model.input_shape[1:],
            'model_path': new_model_path,
            'converted_from': old_model_path
        }
        
        with open('model_info.json', 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        print("모델 정보를 'model_info.json'에 저장했습니다.")
        
        # 백업 생성
        backup_path = old_model_path + '.backup'
        os.rename(old_model_path, backup_path)
        print(f"기존 모델의 백업을 '{backup_path}'로 생성했습니다.")
        
    except Exception as e:
        print(f"변환 중 오류 발생: {str(e)}")
        return

if __name__ == '__main__':
    convert_model() 