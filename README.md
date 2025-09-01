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




   
## 파일구조
```
📦Basic_Connection : 기본 드론 비행 코드
📦drone_education : 드론 교육 실습 코드  
📦Lifter_Tracking : 객체 추적 모델을 드론에 적용한 코드
📦models : 사용하는 모델들
   
