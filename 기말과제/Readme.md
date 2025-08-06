## 딥러닝 심화 프로젝트: 음성인식을 통한 휴대용 장비 제어 시스템

#### **1. 프로젝트 주제 및 소개**

[cite_start]이 프로젝트는 간단한 음성 명령어를 사용하여 휴대용 장비를 제어하는 시스템을 개발하는 것을 목표로 합니다[cite: 4, 28]. [cite_start]산업 현장이나 위험한 환경에서 사용되는 대부분의 휴대용 장비들은 버튼으로 조작됩니다[cite: 33]. [cite_start]그러나 작업자가 장갑을 착용하거나 양손을 다른 장비 조작에 사용하는 경우, 장비 제어에 어려움을 겪을 수 있습니다[cite: 34, 37]. [cite_start]딥러닝 기반 음성 인식 모델이 소형 장비에도 적용할 수 있을 만큼 경량화되면서 이러한 문제를 해결할 수 있는 가능성이 열렸습니다[cite: 35].

[cite_start]본 시스템은 현장 작업자의 편의성을 높이고 [cite: 38][cite_start], 휴대용 장비에 스마트 제어 기능을 추가하여 차세대 경쟁력을 확보하는 데 필요합니다[cite: 39].

---

#### **2. 학습 내용과의 연계성**

* **머신러닝 이론 적용**
    * [cite_start]**딥러닝 모델 설계**: CNN 기반의 음성 인식 모델 아키텍처를 구현했습니다[cite: 45].
    * [cite_start]**특징 추출**: MFCC(Mel-Frequency Cepstral Coefficients) 기법을 사용하여 음성 신호를 전처리했습니다[cite: 48].
    * [cite_start]**모델 훈련**: TensorFlow/Keras를 활용한 지도 학습 기반의 분류 모델을 학습시켰습니다[cite: 49].
* [cite_start]**신호처리 기법**: Mel 필터뱅크를 이용해 인간의 청각 특성을 반영하고 [cite: 51][cite_start], FFT를 통해 시간/주파수 분석을 수행했습니다[cite: 52].
* [cite_start]**임베디드 시스템 최적화**: Arduino Nano RP2040의 SRAM 제약을 고려하여 효율적인 메모리 사용을 위한 학습 모델 경량화를 진행했습니다[cite: 54, 55].

---

#### **3. 시스템 구성 및 구현**

* [cite_start]**하드웨어**: Arduino Nano RP2040 CONNECT를 사용했습니다[cite: 61].
* [cite_start]**데이터셋**: Google Speech Commands Dataset v0.02를 활용했습니다[cite: 63].
* **명령어**:
    * [cite_start]`"marvin"`: 내장 LED가 0.2초 간격으로 5회 깜빡인 후 켜진 상태로 대기합니다[cite: 65].
    * [cite_start]`"on"`: LED가 1초 간격으로 1회 깜빡입니다[cite: 66].
    * [cite_start]`"go"`: LED가 1초 간격으로 2회 깜빡입니다[cite: 69].
    * [cite_start]`"stop"`: LED가 1초 간격으로 3회 깜빡입니다[cite: 70].
    * [cite_start]`"down"`: LED가 1초 간격으로 4회 깜빡입니다[cite: 71].
    * [cite_start]`"happy"`: LED가 꺼지고 슬립 모드로 진입합니다[cite: 72].
    * [cite_start]각 명령어당 약 2000개의 샘플을 사용했습니다[cite: 73].
* [cite_start]**전처리 과정**: 오디오 정규화, MFCC 특징 추출, 그리고 노이즈 및 시간 추가를 통한 데이터 증강을 포함합니다[cite: 75].

---

#### **4. 실험 및 결과 분석**

* **모델 학습**:
    * [cite_start]**최적화**: Adam 옵티마이저 (학습률: 0.001)를 사용했으며 [cite: 93][cite_start], 손실 함수로는 `Categorical Crossentropy`를 사용했습니다[cite: 94]. [cite_start]배치 크기는 32, 에폭은 30으로 설정했습니다[cite: 95, 96].
    * [cite_start]**결과**: 총 30 에폭 동안 학습이 진행되었으며 [cite: 119][cite_start], 최적의 검증 정확도는 24번째 에폭에서 0.9951을 기록했습니다[cite: 120, 121]. [cite_start]최종 테스트 정확도는 0.9928이었습니다[cite: 125]. [cite_start]Early stopping을 통해 과적합을 방지했습니다[cite: 123].
* **모델 경량화**:
    * [cite_start]**최적화 방법**: TensorFlow Lite 모델로 변환하고 양자화(Quantization)를 적용하여 가중치를 압축했습니다[cite: 146, 147, 148].
    * [cite_start]**최적화 결과**: 원본 모델의 크기는 24,621,912 bytes였으나 [cite: 150][cite_start], 최적화된 모델의 크기는 4,077,656 bytes로 감소했습니다[cite: 151].
* **모델 경량화 개선 (추가)**:
    * [cite_start]`stop`과 `down` 명령어에 2배의 가중치를 적용하고, 각 클래스별로 균등한 대표 샘플을 생성하여 클래스별 불균형을 조정했습니다[cite: 211].
    * [cite_start]혼합 정밀도 양자화를 사용하여 중요한 특징 추출 레이어는 FP16을 유지하고, 분류 레이어만 INT8로 양자화했습니다[cite: 211].
