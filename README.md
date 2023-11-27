# 한국어 sLLM 학습 및 평가 프로젝트
## 1. 프로젝트 소개
### 1.1. 프로젝트 목표
- 한국어 sLLM 구축에 필요한 스크립트 제공.
- 한국어 sLLM 구축에 필요한 라이브러리 제공.
- 과제에 따른 언어 프롬프트(Prompt) 생성 모듈 제공.

### 1.2. 프로젝트 구성
- `dataloader/`: 데이터 로더 모듈. 데이터를 불러오고, 전처리하는 기능을 제공(언어 프롬프트 제공).
- `ds_configs/` : DeepSpeed를 이용한 학습을 위한 설정 파일
- `experiments/`: 실험 코드(미완성 코드들이 임시로 위치함).
- `tests/`: 테스트 코드.
- `utils/`: 유틸리티 모듈. collator, tokenizer, callback 등을 위한 기능을 제공.
- `README.md`: 프로젝트 소개 및 사용법 설명.
- `requirements.txt`: 프로젝트 실행에 필요한 라이브러리 목록.
- `build_bpe_tokenizer.py`: BPE 토크나이저 생성 스크립트.
- `build_spm_tokenizer.py`: SentencePiece 토크나이저 생성 스크립트.
- `calculate_num_tokens.py`: 데이터셋의 토큰 개수를 계산하는 스크립트.