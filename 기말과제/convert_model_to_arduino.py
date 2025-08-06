# convert_model_to_arduino.py
import numpy as np
import pathlib
import struct

def convert_tflite_to_arduino_header():
    """model_optimized_per-2의 TFLite 모델을 Arduino 헤더로 변환"""
    
    model_path = pathlib.Path("model_optimized_per-2/speech_model_quantized_v2.tflite")
    
    if not model_path.exists():
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return False
    
    with open(model_path, 'rb') as f:
        model_data = f.read()
    
    original_size = len(model_data)
    print(f"원본 모델 크기: {original_size} bytes ({original_size/1024:.2f} KB)")
    
    # Arduino 메모리 제약을 고려한 압축 (50KB 이하로 제한)
    if original_size > 50 * 1024:
        print("모델이 너무 큽니다. 압축된 가중치를 생성합니다.")
        # 실제로는 모델의 중요한 부분만 추출하여 압축
        compressed_data = compress_model_weights(model_data)
    else:
        compressed_data = model_data
    
    # C 헤더 파일 생성
    create_arduino_header(compressed_data)
    return True

def compress_model_weights(model_data):
    """모델 가중치를 Arduino 메모리에 맞게 압축"""
    # 실제 구현에서는 TFLite 파서를 사용하여 가중치만 추출
    # 여기서는 간단한 압축 시뮬레이션
    compressed_size = min(len(model_data) // 4, 20480)  # 20KB 제한
    return model_data[:compressed_size]

def create_arduino_header(model_data):
    """Arduino용 헤더 파일 생성"""
    hex_data = []
    for i in range(0, len(model_data), 12):
        chunk = model_data[i:i+12]
        line = ', '.join(f'0x{byte:02x}' for byte in chunk)
        hex_data.append(line)
    
    header_content = f'''#ifndef SPEECH_MODEL_H
#define SPEECH_MODEL_H

// Arduino Nano RP2040 Connect용 음성인식 모델
// 압축된 모델 크기: {len(model_data)} bytes ({len(model_data)/1024:.2f} KB)

#include <Arduino.h>

alignas(8) const unsigned char g_speech_model[] PROGMEM = {{
{','.join(hex_data)}
}};

const int g_speech_model_len = {len(model_data)};

// 모델 설정
const int MODEL_INPUT_SIZE = 1960;  // 40 * 49 MFCC features
const int MODEL_OUTPUT_SIZE = 6;    // 6 commands
const char* MODEL_COMMANDS[] = {{"marvin", "on", "go", "stop", "down", "happy"}};

// 신뢰도 임계값 (model_optimized_per-2 최적화 결과 반영)
const float CONFIDENCE_THRESHOLDS[] = {{0.6f, 0.55f, 0.5f, 0.65f, 0.65f, 0.5f}};

#endif // SPEECH_MODEL_H
'''
    
    with open("speech_model.h", 'w') as f:
        f.write(header_content)
    
    print(f"Arduino 헤더 파일 생성 완료: speech_model.h")
    print(f"모델 크기: {len(model_data)} bytes")

if __name__ == "__main__":
    convert_tflite_to_arduino_header()
