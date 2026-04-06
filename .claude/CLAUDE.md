# Content Repurposer Harness

1개 원본 콘텐츠를 블로그·SNS·뉴스레터·프레젠테이션·스크립트로 다중 변환하는 에이전트 팀 하네스.

## 구조

```
.claude/
├── agents/
│   ├── source-analyst.md        — 원본 분석가 (구조 분석, 핵심 추출, 변환 전략)
│   ├── blog-writer.md           — 블로그 작가 (SEO 최적화 블로그 포스트)
│   ├── sns-copywriter.md        — SNS 카피라이터 (플랫폼별 포스트)
│   ├── presentation-builder.md  — 프레젠테이션 빌더 (슬라이드 구성)
│   └── quality-reviewer.md      — 품질 검증자 (교차 검증, 메시지 일관성)
├── skills/
│   ├── content-repurposer/
│   │   └── skill.md             — 오케스트레이터 (팀 조율, 워크플로우, 에러핸들링)
│   ├── platform-adaptation/
│   │   └── skill.md             — sns-copywriter+blog-writer 확장 (플랫폼별 DNA, 변환 매트릭스)
│   └── content-atomization/
│       └── skill.md             — source-analyst+presentation-builder 확장 (MINE 분석, 원자 분류)
└── CLAUDE.md                    — 이 파일
```

## 사용법

`/content-repurposer` 스킬을 트리거하거나, "이 콘텐츠 리퍼포징해줘" 같은 자연어로 요청한다.

## 산출물

모든 산출물은 `_workspace/` 디렉토리에 저장된다:
- `00_input.md` — 사용자 입력 정리
- `01_source_analysis.md` — 원본 분석 보고서
- `02_blog_post.md` — 블로그 포스트
- `03_sns_package.md` — SNS 포스트 패키지
- `04_presentation.md` — 프레젠테이션 슬라이드
- `05_review_report.md` — 리뷰 보고서

## 답변 언어

모든 답변은 한국어로 한다.

---

# Velog 공부기록 시리즈 — AI 입문 강의

`ai_lecture-master/` 강의 자료를 기반으로 day별 Velog 글을 작성하는 워크플로우.

## 강의 목록

| day | 디렉토리 | 주제 | 노트북 | 글 상태 |
|-----|----------|------|--------|---------|
| 1 | `day1_Python_for_AI` | Python 기초 + NumPy + Pandas | `0_1_AI를_위한_Python_(Easy).ipynb` | ✅ 완료 |
| 2 | `day2_AI_Math` | 선형회귀 · 로지스틱회귀 (NumPy 구현) | `0_2_선형회귀와_로지스틱회귀_(Easy).ipynb` | ✅ 완료 |
| 3 | `day3_AI_Data_EDA` | EDA + Scikit-learn 파이프라인 | `1_1_EDA와_Scikit_learn_(Easy).ipynb` | ⬜ 미작성 |
| 4 | `day4_MLP` | 신경망 + MLP (PyTorch) | `1_2_신경망(Neural_Network)과_MLP_(Easy).ipynb` | ⬜ 미작성 |
| 5 | `day5_CNN` | CNN + Fine-tuning | `3_1_CNN_모델과_Fine_tuning_(Easy).ipynb` | ⬜ 미작성 |
| 6 | `day6_llm` | 파운데이션 모델 → 멀티모달 · 디퓨전 | `3_2_파운데이션에서_멀티모달까지,_LLM_모델의_진화(Easy).ipynb` | ⬜ 미작성 |
| 7 | `day7_embeddings` | 토큰화 + 임베딩 + Transformer | `2_1_토큰화_임베딩_(Easy).ipynb` | ⬜ 미작성 |
| 8 | `day8_data_synthetic` | LLM 데이터 생성 · 증강 | `2_2_AI를_이용한_데이터_생성법(Easy)_Solar.ipynb` | ⬜ 미작성 |
| 10 | `day10_rag` | LangChain + RAG | `4-1_LangGraph와 RAG_r2.ipynb` 외 | ⬜ 미작성 |
| 11 | `day11_agent` | Multi-turn + AI Agent + Gradio | `4-2(1)_LangGraph기본_r1.ipynb` 외 | ⬜ 미작성 |
| 12 | `day12_PEFT` | PEFT · LoRA · QLoRA | `5_1_효율적인_Fine_tuning_PEFT(Easy).ipynb` | ⬜ 미작성 |
| 14 | `day14_Quantization` | 모델 양자화 + llama.cpp | `5-2_양자화모델_성능측정_with_llama.cpp(Easy)_r7.ipynb` | ⬜ 미작성 |

## 글 작성 워크플로우

새 day 글을 요청할 때마다 아래 순서로 진행한다.

### 1단계 — 노트북 내용 추출

노트북 파일이 256KB를 초과하는 경우 Read 툴 대신 Bash로 파싱한다:

```bash
python3 -c "
import json, sys
nb = json.load(open('경로/노트북.ipynb', encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    t = cell['cell_type']
    src = ''.join(cell['source'])
    print(f'[{i}] {t}: {src[:200]}')
    print('---')
" | head -300
```

### 2단계 — blog-writer 에이전트로 초안 작성

- 산출물 저장: `_workspace/02_blog_post.md` (덮어쓰기)
- 프론트매터: `title`, `tags`, `date` 포함
- 목표 분량: 2,000~3,000자
- 톤: 공부 기록 (딱딱하지 않게, 정확하게)
- 구성: 들어가며 → 핵심 개념별 섹션 → 마무리(요약 표 + 다음 day 예고)

### 3단계 — 썸네일 생성

- 스크립트: `_workspace/dayN_gen_thumbnail.py` 수정 후 실행
- 출력: `_workspace/dayN_thumbnail.png` (1200×630px)
- **day별 변경 항목**:
  - `icons` 리스트 — 아이콘 박스 텍스트 (예: `["Py", "Np", "Pd"]`)
  - `day_txt` — `"Day N"` 텍스트
  - `title` — 메인 타이틀
  - `subtitle` — 서브타이틀
  - `tags` 리스트 — pill 형태 태그 배지

### 4단계 — 완료 후 강의 목록 상태 업데이트

이 파일의 강의 목록 표에서 해당 day의 `⬜ 미작성` → `✅ 완료`로 변경한다.

## 썸네일 디자인 공통 스펙

| 항목 | 값 |
|------|-----|
| 크기 | 1200 × 630 px |
| 배경 | 대각선 그라디언트 `#1a73e8` → `#4285f4` → `#34a853` (135도) |
| 레이아웃 | 세로 중앙 정렬: 아이콘 박스 → 시리즈 라벨 → Day번호 → 메인타이틀 → 서브타이틀 → 태그 |
| 아이콘 박스 | 80×80px, 반투명 흰색 (`rgba(255,255,255,0.2)`), 둥근 모서리 |
| 태그 배지 | pill 형태, 반투명 흰색 배경, 흰색 테두리 |
| 배경 패턴 | 반투명 흰색(10%) 기하 도형 4개 (사각형·원·직사각형·채워진 사각형) |
| 폰트 | Malgun Gothic (Windows 기본, `C:/Windows/Fonts/malgun*.ttf`) |
| 스크립트 위치 | `_workspace/dayN_gen_thumbnail.py` |

## 산출물 파일 위치

| 파일 | 경로 |
|------|------|
| 블로그 포스트 | `_workspace/dayN_blog_post.md` |
| 썸네일 이미지 | `_workspace/dayN_thumbnail.png` |
| 썸네일 생성 스크립트 | `_workspace/dayN_gen_thumbnail.py` |
| 원본 강의 디렉토리 | `C:/Users/user/Desktop/_BLOG_POST/ai_lecture-master/` |