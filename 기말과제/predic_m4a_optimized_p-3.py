import tensorflow as tf
import numpy as np
import os
import json
import pathlib
from pydub import AudioSegment
from tqdm import tqdm

class OptimizedModelTester:
    def __init__(self, model_dir='model_optimized_per-3'):
        """경량화된 모델 테스터 초기화"""
        self.model_dir = pathlib.Path(model_dir)
        self.interpreter = None
        self.commands = None
        self.input_details = None
        self.output_details = None
        self.load_optimized_model()
    
    def load_optimized_model(self):
        """최적화된 TFLite 모델 로드"""
        model_path = self.model_dir / 'speech_model_quantized_v3.tflite'
        
        if not model_path.exists():
            raise FileNotFoundError(f"최적화된 모델을 찾을 수 없습니다: {model_path}")
        
        # TFLite 인터프리터 초기화
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        
        # 입출력 텐서 정보 가져오기
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 모델 정보 로드
        info_path = self.model_dir / 'optimization_info_v3.json'
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        
        self.commands = model_info['original_commands']
        
        print("최적화된 모델 로드 완료:")
        print(f"- 명령어: {self.commands}")
        print(f"- 입력 형태: {self.input_details[0]['shape']}")
        print(f"- 입력 타입: {self.input_details[0]['dtype']}")
        print(f"- 출력 형태: {self.output_details[0]['shape']}")
        print(f"- 모델 크기: {model_info['model_size_bytes']} bytes")
    
    def preprocess_m4a_for_tflite(self, file_path):
        """M4A 파일을 TFLite 모델 입력 형식으로 전처리"""
        try:
            # M4A 파일 로드 및 변환
            audio = AudioSegment.from_file(file_path, format="m4a")
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # numpy 배열로 변환
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.sample_width == 2:
                samples = samples / 32768.0
            
            # 1초 길이로 조정
            target_samples = 16000
            if len(samples) > target_samples:
                samples = samples[:target_samples]
            else:
                samples = np.pad(samples, (0, target_samples - len(samples)), mode='constant')
            
            # 스펙트로그램 생성
            audio_tensor = tf.constant(samples, dtype=tf.float32)
            spectrogram = tf.signal.stft(audio_tensor, frame_length=255, frame_step=128)
            spectrogram = tf.abs(spectrogram)
            
            # 입력 형태에 맞게 조정
            input_shape = self.input_details[0]['shape']
            spectrogram = tf.image.resize(spectrogram[..., tf.newaxis], 
                                        [input_shape[1], input_shape[2]])
            
            # 정규화 (필요시)
            spectrogram = (spectrogram - tf.reduce_mean(spectrogram)) / tf.math.reduce_std(spectrogram)
            
            # 배치 차원 추가
            spectrogram = tf.expand_dims(spectrogram, 0)
            
            # 데이터 타입 변환 (INT8인 경우)
            if self.input_details[0]['dtype'] == np.int8:
                input_scale, input_zero_point = self.input_details[0]['quantization']
                spectrogram = spectrogram / input_scale + input_zero_point
                spectrogram = tf.cast(tf.round(spectrogram), tf.int8)
            
            return spectrogram.numpy()
            
        except Exception as e:
            print(f"전처리 오류 ({file_path}): {str(e)}")
            return None
    
    def predict_m4a_file(self, file_path):
        """단일 M4A 파일 예측"""
        features = self.preprocess_m4a_for_tflite(file_path)
        if features is None:
            return None
        
        try:
            # 입력 설정
            self.interpreter.set_tensor(self.input_details[0]['index'], features)
            
            # 추론 실행
            self.interpreter.invoke()
            
            # 결과 가져오기
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # 출력이 양자화된 경우 역양자화
            if self.output_details[0]['dtype'] == np.int8:
                output_scale, output_zero_point = self.output_details[0]['quantization']
                output = output_scale * (output.astype(np.float32) - output_zero_point)
            
            predicted_index = np.argmax(output[0])
            confidence = output[0][predicted_index]
            predicted_command = self.commands[predicted_index]
            
            return {
                'file_path': file_path,
                'file_name': pathlib.Path(file_path).name,
                'predicted_command': predicted_command,
                'confidence': float(confidence),
                'predicted_index': int(predicted_index),
                'all_probabilities': {
                    self.commands[i]: float(output[0][i]) for i in range(len(self.commands))
                }
            }
            
        except Exception as e:
            print(f"추론 오류 ({file_path}): {str(e)}")
            return None
    
    def test_testsound_folder(self, folder_path='TestSound'):
        """TestSound 폴더의 모든 M4A 파일 테스트"""
        test_path = pathlib.Path(folder_path)
        
        if not test_path.exists():
            print(f"TestSound 폴더를 찾을 수 없습니다: {folder_path}")
            return
        
        m4a_files = list(test_path.rglob('*.m4a'))
        
        if not m4a_files:
            print(f"M4A 파일을 찾을 수 없습니다: {folder_path}")
            return
        
        print(f"\n경량화된 모델로 {len(m4a_files)}개 M4A 파일 테스트")
        print("="*70)
        
        results = []
        
        for m4a_file in tqdm(m4a_files, desc="파일 처리"):
            result = self.predict_m4a_file(str(m4a_file))
            
            if result:
                results.append(result)
                
                print(f"\n파일: {result['file_name']}")
                print(f"예측 명령어: {result['predicted_command']}")
                print(f"신뢰도: {result['confidence']:.4f}")
                
                # 상위 6개 확률 표시
                sorted_probs = sorted(result['all_probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)[:6]
                print("상위 6개 확률:")
                for cmd, prob in sorted_probs:
                    print(f"  - {cmd}: {prob:.4f}")
                print("-" * 50)
        
        if results:
            self.print_test_statistics(results)
    
    def print_test_statistics(self, results):
        """테스트 결과 통계"""
        print(f"\n{'='*70}")
        print("경량화 모델 테스트 결과:")
        print(f"총 테스트 파일: {len(results)}")
        
        # 명령어별 분포
        command_counts = {}
        for result in results:
            cmd = result['predicted_command']
            command_counts[cmd] = command_counts.get(cmd, 0) + 1
        
        print("\n예측된 명령어 분포:")
        for cmd in self.commands:
            count = command_counts.get(cmd, 0)
            percentage = (count / len(results)) * 100 if results else 0
            print(f"  - {cmd}: {count}개 ({percentage:.1f}%)")
        
        # 신뢰도 분석
        confidences = [r['confidence'] for r in results]
        print(f"\n신뢰도 분석:")
        print(f"  - 평균: {np.mean(confidences):.4f}")
        print(f"  - 최대: {np.max(confidences):.4f}")
        print(f"  - 최소: {np.min(confidences):.4f}")

def main():
    """메인 실행 함수"""
    
    print("\n경량화된 모델로 M4A 파일 테스트")
    tester = OptimizedModelTester()
    tester.test_testsound_folder('TestSound')

if __name__ == "__main__":
    main()
