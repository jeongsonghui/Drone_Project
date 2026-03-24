# 🚀 Drone_Project  

## 👥 Team Members  
- **김희지**  
- **정송희**  

## 📌 수행 중인 과제  
### 1️⃣ Object Detection  
🔹 **목적**: 드론을 활용하여 서로의 얼굴을 감지하고 구분하기  
🔹 **모델**: YOLO v10  
🔹 **드론**: DJI Tello Drone  
🔹 **데이터셋**: Roboflow를 사용하여 각자의 얼굴 100장씩 수집 (총 200장)  
🔹 **클래스**: `{ "heeji", "songhee" }`  

  ### 📊 성능 평가  
  | Metric   | Score  |  
  |----------|--------|  
  | **Precision** | 0.9914 |  
  | **Recall** | 0.9611 |  
  | **mAP@50** | 0.9917 |  
  | **mAP@50-95** | 0.7346 |  

### 2️⃣: 연구주제  
🔹 **목적**: 물류창고나 건설 현장에서 리프터(중장비) 안전 관리가 중요한데, 사람이 직접 모니터링하는 방식은 비효율적이었습니다. 이를 해결하기 위해 드론이 실시간으로 리프터를 감지·추적하여 위험 상황을 모니터링할 수 있는 시스템을 제안했습니다.  
🔹 **모델**: YOLO v10 + DeepSORT  
🔹 **드론**: DJI Tello Drone   
🔹 **데이터셋**: ExDark 데이터셋의 'person' 클래스와 드론 카메라로 직접 수집한 Lifter 데이터 (총 600장)  
🔹 **클래스**: `{ "person", "lifter" }`  


 ### 📊 성능 평가  
  | Metric   | Score  |  
  |----------|--------|  
  | **Precision** | 0.7383 |  
  | **Recall** | 0.7519 |  
  | **mAP@50** | 0.7543 |  
  | **mAP@50-95** | 0.5704 |  
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c5f9845e-d1e3-4a01-a558-8461da914d91" />

<img width="1786" height="906" alt="스크린샷 2025-09-02 012330" src="https://github.com/user-attachments/assets/313afb86-0bb9-4a56-8dac-03ea17abe36a" />

### 3️⃣: Lifter Tracking 자율주행 드론 성능
📌 **실행 환경**
- 드론: Tello Robomaster  
- 모델: YOLO v10 + DeepSORT 
- 입력 해상도: 480 × 640  
- 개발 언어/프레임워크: Python, PyTorch 

📊 **성능 결과**
| 항목 | 값 |
|------|------|
| IO FPS | 79 fps |
| Inference FPS | 7 fps |
| 평균 정확도 | 0.56 (Lifter) |

🎯 **객체 탐지 및 추적**
- 탐지 대상: Lifter, 사람  
- 탐지 결과 기반 객체 위치 추적 수행  
- 화면 중앙 유지 + 위험 요소(리프터) 감지 시 경고 발생  

🚁 **제어 동작**
- 드론은 탐지된 객체 중심 좌표 기반으로 실시간 이동 제어 (`rc` 명령어 활용)  
- 로그 기록: 매 프레임 FPS, 탐지 결과, 타겟 좌표 등  

⚡ **개선 사항**
- LifoQueue를 사용하여 항상 최신 프레임만 추론하도록 구조 개선  
- Queue가 가득 찰 경우 이전 프레임 제거 전략 적용 → 지연(latency) 누적 방지  
- 초기 FPS: 3 → 최적화 후 Inference FPS 7로 실시간에 가까운 처리 가능  
- 안정적인 객체 추적 및 경고 시스템 구현



   
## 파일구조
```
📦 Basic_Connection  
 ┗ 기본 드론 비행 코드  

📦 drone_education  
 ┗ 드론 교육 실습 코드  

📦 Experiment/ 
 ┗ OSL 등 다양한 방법으로 모델 실험 및 자율 주행 경로(Shortest_path) 테스트

📦 Images
 ┗ 리프터와 어두운 곳에서의 사람 이미지 데이터셋

📦 Lifter_Tracking  
 ┗ 객체 추적 모델을 드론에 적용한 코드  
    ┗ 📄 lifter_autofollow.py  
       ┗ 리프터 및 사람을 감지하고, 리프터를 자동으로 추적하여 화면 중앙에 유지
📦 models   
 ┗ 학습된 모델들

📦 YOLO10_finetuning(drone).ipynb  
 ┗ YOLOv10 파인튜닝 및 학습 과정 노트북 

```

---

📓 **YOLO10_finetuning(drone) Notebook**

- 🔍 [nbviewer로 보기](https://nbviewer.org/github/jeongsonghui/Drone_Project/blob/main/YOLO10_finetuning(drone).ipynb)

## 실행방법

1. DJI Tello RoboMaster 드론 전원 ON  
2. PC에서 드론 Wi-Fi에 연결  
3. 프로젝트 루트 디렉토리에서 아래 명령어 실행  

```bash
cd Lifter_Tracking
python lifter_autofollow.py

```

### 4️⃣ Extension: Drone Coding Education

본 프로젝트를 기반으로, 드론 제어 및 AI 모델 활용을 학습할 수 있는 **드론 코딩 교육 자료**를 제작했습니다.

### 🎯 목적
- 파이썬이 가능한 중고등학생들을 대상으로 ML, 컴퓨터 비전의 이해와 드론을 조작하는 코딩법을 익히게 하기 위함.

### 📖 내용 구성
- 드론에 대한 소개, 코딩을 통한 간단 조작법, CV 모델 학습, 모델을 드론에 적용시키기 

