# run_batch_evaluation.py 사용 가이드

## 개요

`run_batch_evaluation.py` 스크립트는 디렉토리 내 학생들이 제출한 모델 체크포인트(`student_checkpoint.pth`, `student_improved_checkpoint.pth`)를 로드 후 평가하고, 그 결과를 CSV 파일로 정리합니다. 

**주요 기능:**
- `student_checkpoint.pth`(baseline), `student_improved_checkpoint.pth`(improved) 모델에 대해 CIFAR-100 테스트 데이터셋으로 정확도 평가
- 결과를 `student_accuracy_evaluation.csv` 파일로 저장

## 필요 환경

- Python 3.9 이상
- 다음 패키지 설치 필요:
  - `torch`, `torchvision`
  - `gdown`
  - `tqdm`

```bash
$ git clone https://github.com/hushon/ee837_eval
```

## 디렉토리 구조 예시

KLMS 에서 제출물 파일을 다운받은 후 아래 구조처럼 레포지토리 안에 압축 해제. 

```
ee837_eval/
    ├── README.md
    ├── run_batch_evaluation.py
    ├── 20243113(김병철)_5242360_assignsubmission_file_/
    │   ├── student_checkpoint.pth
    │   └── student_improved_checkpoint.pth (옵션)
    ├── 20243119(김상윤)_5242322_assignsubmission_file_/
    │   └── student_checkpoint.pth
    └── ...
```

  - 각 학생별 폴더 이름은 `{학번}({이름})_xxx_assignsubmission_file_` 형태를 갖는다고 가정.
  - `student_checkpoint.pth`: 기본 모델 제출 체크포인트
  - `student_improved_checkpoint.pth`: 개선된 모델 제출 체크포인트 (옵션)
  - 만약 어떤 학생의 체크포인트 파일이 없거나 파일명이 매칭되지 않으면 결과 칼럼은 공백으로 남음.

## 실행 방법

```bash
$ python run_batch_evaluation.py ./ 
```

위 명령은 현재 디렉토리(`./`)를 기준으로 학생들의 제출 폴더를 탐색한 뒤, 모든 학생의 모델을 평가한다. 처리 결과는 `./student_accuracy_evaluation.csv` 파일로 저장된다.

## 출력 결과

`student_accuracy_evaluation.csv` 예시:

|student_id|student_name|baseline_acc|improved_KD|
|---|---|---|---|
|20243113|김병철|42.5|45.7|
|20243119|김상윤|40.1| |