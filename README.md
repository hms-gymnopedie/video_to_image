# Video → Image AI Pipeline (V4.0.0)

A dashboard for turning videos into reconstruction-ready image sequences for
3D Gaussian Splatting (3DGS), 2D Gaussian Splatting (2DGS), and COLMAP / SfM.

It combines FFmpeg-based frame extraction with Laplacian-variance blur filtering
and a full **SAM 2.1 + Grounded-SAM 2** masking suite: point prompts, one-click
video propagation, and text-prompt batch masking — all saved as binary masks
(255 = keep, 0 = exclude) that drop directly into standard reconstruction
pipelines.

- Backend: FastAPI + PyTorch (SAM 2.1, Grounding DINO)
- Frontend: React 19 + Vite + MUI 9
- Runtime: macOS (MPS), CUDA, or CPU

---

## Features

### Extraction & quality filtering
- Drag-and-drop video upload with metadata probe (resolution, FPS, duration, aspect).
- FFmpeg frame extraction at a user-chosen FPS.
- Per-frame Laplacian-variance blur score.
- Auto-split into `sharp/` and `blur/` folders with a live threshold slider.
- Pipeline presets (3DGS / 2DGS / COLMAP) that configure FPS + threshold in one click.
- **Manual reclassification** — override any frame between sharp/blur from the editor
  (SHARP / BLUR button group); the file is physically moved.
- **Keyboard navigation** — `←` / `→` cycle through frames inside the mask editor.

### AI masking (SAM 2.1 + Grounded-SAM 2)
- **Point-prompt masking** (Phase 1) — click the object in any frame; SAM 2.1 returns
  the instance mask.
- **Video propagation** (Phase 2) — one click on *any* frame propagates a temporally
  consistent mask across the full sequence via the SAM 2.1 video predictor.
- **Text-prompt masking** (Phase 4, Grounded-SAM 2) — prompt with comma-separated
  phrases like `person, car, sky`; Grounding DINO detects, SAM 2.1 segments.
- **Reconstruction-ready binary masks** (Phase 3) — `<scene>/masks/<frame>.png`,
  255=keep / 0=exclude, compatible with 3DGS, 2DGS, COLMAP.
- **COMBINE mode** — ANDs new masks with existing ones, so propagation + text
  prompts on the same scene never overwrite each other.
- **skip_empty** — frames where nothing is detected never write a file, keeping
  the `masks/` folder free of no-op outputs.
- **Mask preview** — per-frame red-tinted overlays in a paginated grid for
  visual verification; auto-opens after propagation / text-batch completes.
- Dynamic model switching between Hiera-Tiny / Small / Base+ / Large.
- Apple Silicon (MPS) and CUDA auto-detected; CPU fallback for development.

---

## Prerequisites

- Node.js ≥ 20 & npm
- Python 3.11+
- FFmpeg on `PATH`
- (GPU optional) CUDA device, or Apple Silicon (M-series) for MPS

---

## Installation

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
# SAM 2 has no PyPI package — install from source:
pip install "git+https://github.com/facebookresearch/sam2.git"
```

Model weights are downloaded automatically on first run via the Hugging Face
Hub (`facebook/sam2.1-hiera-*` and `IDEA-Research/grounding-dino-base`) into
`~/.cache/huggingface/hub/`. No manual weight placement is required.

### Frontend

```bash
cd frontend
npm install
```

---

## Running

### 1. Backend

```bash
cd backend
source venv/bin/activate
python main.py
```

Serves at `http://localhost:8080`. The server starts immediately; SAM 2.1
(`base_plus` by default) loads in the background. A status indicator in the
dashboard reflects `SAM2_ONLINE` / `SAM2_OFFLINE`.

### 2. Frontend

```bash
cd frontend
npm run dev
```

Opens at `http://localhost:5173`.

---

## Typical workflow

1. Drop a video into `[01] INPUT_SOURCE`.
2. Pick a preset (3DGS / 2DGS / COLMAP) or set FPS + blur threshold manually.
3. Choose an output folder from the tree (or keep the default).
4. Press **START_PIPELINE** → frames are extracted, scored, and split into
   `sharp/` and `blur/`.
5. (Optional) Click a thumbnail to enter the **AI MASKING EDITOR**, click
   objects to set points, then press **PROPAGATE TO SEQUENCE** for per-frame
   masks. Use `←` / `→` to cycle frames, SHARP / BLUR to reclassify.
6. (Optional) In `[04] RECONSTRUCTION_MASKING`, enter text queries
   (`person, car`) and run **TEXT_PROMPT_MASK** for an additional pass on the
   `sharp/` folder. With COMBINE enabled, the two passes are unioned as
   "things to exclude".
7. Press **PREVIEW MASKS** (auto-opens after masking completes) to verify
   every overlay.
8. Point your 3DGS / 2DGS / COLMAP run at `<scene>/sharp/` for images and
   `<scene>/masks/` for masks.

---

## Binary mask convention

```
<scene>/
├── sharp/                f_0001.jpg, f_0002.jpg, …
├── blur/
├── masks/                f_0001.png, f_0002.png, …   (255 = keep, 0 = exclude)
└── mask_preview/         red-tinted overlays for UI verification
```

| Pipeline | Frames dir      | Masks dir        | Notes                                      |
|----------|-----------------|------------------|--------------------------------------------|
| 3DGS     | `sharp/`        | `masks/`         | `<basename>.png`                           |
| 2DGS     | `sharp/`        | `masks/`         | `<basename>.png`                           |
| COLMAP   | `sharp/`        | `masks/`         | Rename to `<full_name>.png` if required    |

---

## Main API endpoints

| Method | Path                           | Purpose                                   |
|--------|--------------------------------|-------------------------------------------|
| GET    | `/health`                      | Version, device, SAM2 / DINO ready flags  |
| POST   | `/upload`                      | Multipart video upload                    |
| GET    | `/metadata/{file_id}`          | FFprobe metadata                          |
| WS     | `/ws/process/{file_id}`        | Extract + blur-score pipeline             |
| POST   | `/change-model/{size}`         | Switch SAM 2.1 size                       |
| POST   | `/segment`                     | Phase 1 — point-prompt single-frame       |
| WS     | `/ws/segment-video`            | Phase 2 — video propagation               |
| POST   | `/segment-text`                | Phase 4 — text-prompt single-frame        |
| WS     | `/ws/segment-text-batch`       | Phase 4 — text-prompt batch               |
| POST   | `/reconstruction-mask-info`    | Phase 3 — mask path + count               |
| POST   | `/generate-mask-previews`      | Build red-tint overlay grid               |
| GET    | `/overlay-file?path=…`         | Serve an overlay/mask file by abs path    |
| POST   | `/reclassify`                  | Move a frame between `sharp/` and `blur/` |
| GET    | `/directories`                 | Output directory tree                     |
| POST   | `/create-directory`            | Create a subfolder                        |
| POST   | `/sync-folders`                | Re-apply threshold to existing results    |

---

## Project layout

```
backend/
├── main.py             FastAPI app, all endpoints
├── ai_engine.py        SAMEngine class + mask I/O helpers
├── requirements.txt
├── output/             Extracted frames (sharp/ + blur/) and masks/
└── masks/preview/      Preview overlays for point-prompt segmentation

frontend/
├── src/App.tsx         Single-page dashboard
└── package.json
```

---

## Model notes

- SAM 2.1 sizes: **Tiny** (fastest) → **Small** → **Base+** (recommended) → **Large** (best).
- Legacy model names `vit_b` / `vit_l` / `vit_h` are accepted and mapped to
  `tiny` / `base_plus` / `large` for backward compatibility.
- Grounding DINO is loaded lazily on the first text-prompt call.
- A PyTorch 2.6 monkeypatch forces `weights_only=False` for SAM 2 checkpoint
  loading.

---

## Troubleshooting

- **`No module named 'sam2'`** — run the SAM 2 git install line above; it is
  not available on PyPI.
- **`invalid literal for int(): 'f_0033'`** — resolved in V4.0.0. The video
  predictor now stages integer-named symlinks in a temp dir, so extraction
  filenames like `f_0001.jpg` work unchanged.
- **`TypeError: … got an unexpected keyword argument 'box_threshold'`** —
  resolved by a try/fallback shim that also supports `threshold=` in
  `transformers ≥ 4.51`.
- **Preview grid empty** — verify that `<scene>/masks/` contains files; check
  the backend log for `[PREVIEW] scene=… items=N`. If the scene lives outside
  `OUTPUT_DIR`, files are served via `/overlay-file?path=…` instead of
  `/images/…`.
- **Masks from two passes overwriting each other** — keep COMBINE checked in
  the `[04]` header so each run ANDs with the previous one.

See `HISTORY.md` for the full version log.

---

# (Korean / 한국어)

# 영상 → 이미지 AI 파이프라인 (V4.0.0)

3D Gaussian Splatting(3DGS), 2D Gaussian Splatting(2DGS), COLMAP/SfM 재구성에
바로 투입할 수 있는 이미지 시퀀스와 바이너리 마스크를 생성하는 대시보드입니다.

FFmpeg 기반 프레임 추출 + Laplacian 분산 블러 필터링에, **SAM 2.1 + Grounded-SAM 2**
마스킹(포인트 · 동영상 전파 · 텍스트 프롬프트)을 결합하여 `masks/<frame>.png`
(255=유지, 0=제외) 포맷으로 바로 내보냅니다.

- 백엔드: FastAPI + PyTorch (SAM 2.1, Grounding DINO)
- 프론트엔드: React 19 + Vite + MUI 9
- 실행 환경: macOS(MPS) / CUDA / CPU

---

## 주요 기능

### 추출 · 품질 필터링
- 드래그 앤 드롭 업로드 + 메타데이터(해상도/FPS/길이/비율) 자동 추출.
- 사용자 지정 FPS로 FFmpeg 프레임 추출.
- 프레임별 Laplacian 분산 기반 블러 점수 계산.
- 임계값 슬라이더로 `sharp/` · `blur/` 자동 분류.
- 재구성 프리셋(3DGS / 2DGS / COLMAP) — FPS·임계값을 한 번에 설정.
- **수동 재분류** — 에디터에서 SHARP / BLUR 버튼으로 프레임을 물리적으로 이동.
- **키보드 이동** — 마스크 에디터에서 `←` / `→`로 프레임 순회.

### AI 마스킹 (SAM 2.1 + Grounded-SAM 2)
- **포인트 프롬프트 마스킹** (Phase 1) — 객체를 클릭하면 SAM 2.1이 인스턴스
  마스크를 반환.
- **동영상 전파** (Phase 2) — 아무 프레임에서 한 번 클릭하면 SAM 2.1
  비디오 프레딕터가 시퀀스 전체에 시간적으로 일관된 마스크를 생성.
- **텍스트 프롬프트 마스킹** (Phase 4, Grounded-SAM 2) — `person, car, sky`
  처럼 쉼표로 구분된 쿼리 입력 → Grounding DINO 탐지 → SAM 2.1 세그먼트.
- **재구성용 바이너리 마스크** (Phase 3) — `<scene>/masks/<frame>.png`
  (255=유지 / 0=제외), 3DGS / 2DGS / COLMAP 호환.
- **COMBINE 모드** — 새 마스크를 기존 마스크와 AND 연산으로 결합.
  전파 + 텍스트 프롬프트를 같은 씬에 순차 실행해도 덮어쓰기 없음.
- **skip_empty** — 검출이 없는 프레임은 파일을 생성하지 않아 `masks/`
  폴더에 의미 없는 파일이 쌓이지 않음.
- **마스크 프리뷰** — 프레임별 빨강 오버레이를 페이지네이션 그리드로 표시.
  전파·배치 완료 시 자동 오픈.
- 모델 동적 전환: Hiera-Tiny / Small / Base+ / Large.
- Apple Silicon(MPS) · CUDA 자동 감지, CPU 폴백.

---

## 사전 준비

- Node.js ≥ 20 + npm
- Python 3.11+
- FFmpeg가 `PATH`에 등록
- (선택) CUDA GPU 또는 Apple Silicon (M 시리즈)

---

## 설치

### 백엔드

```bash
cd backend
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
# SAM 2는 PyPI 미등록 — git 설치 필수:
pip install "git+https://github.com/facebookresearch/sam2.git"
```

모델 가중치는 첫 실행 시 Hugging Face Hub에서 자동 다운로드됩니다
(`facebook/sam2.1-hiera-*`, `IDEA-Research/grounding-dino-base` →
`~/.cache/huggingface/hub/`). 별도 파일 배치 작업은 필요 없습니다.

### 프론트엔드

```bash
cd frontend
npm install
```

---

## 실행

### 1. 백엔드

```bash
cd backend
source venv/bin/activate
python main.py
```

`http://localhost:8080`에서 실행됩니다. 서버는 즉시 기동되며 SAM 2.1
(`base_plus` 기본값)은 백그라운드 로딩. 대시보드의 `SAM2_ONLINE` / `SAM2_OFFLINE`
인디케이터로 상태 확인.

### 2. 프론트엔드

```bash
cd frontend
npm run dev
```

`http://localhost:5173`에서 실행.

---

## 표준 워크플로우

1. `[01] INPUT_SOURCE`에 동영상 드롭.
2. 프리셋 선택(3DGS / 2DGS / COLMAP) 또는 FPS·임계값 직접 설정.
3. 트리에서 출력 폴더 선택(기본값 사용도 가능).
4. **START_PIPELINE** → 프레임 추출 + 블러 점수 + `sharp/` / `blur/` 분류.
5. (선택) 썸네일 클릭 → **AI MASKING EDITOR** → 객체 클릭으로 포인트 설정
   → **PROPAGATE TO SEQUENCE**로 전체 시퀀스에 마스크 전파.
   `←` / `→` 로 프레임 이동, SHARP / BLUR 로 재분류.
6. (선택) `[04] RECONSTRUCTION_MASKING`에서 텍스트 쿼리(`person, car`)
   입력 후 **TEXT_PROMPT_MASK**로 `sharp/` 폴더 일괄 처리. COMBINE 체크 시
   두 패스의 제외 영역이 합쳐짐.
7. **PREVIEW MASKS**(마스킹 완료 시 자동 오픈)로 오버레이 검증.
8. 3DGS / 2DGS / COLMAP 에서 이미지는 `<scene>/sharp/`, 마스크는
   `<scene>/masks/`로 지정.

---

## 바이너리 마스크 구조

```
<scene>/
├── sharp/                f_0001.jpg, f_0002.jpg, …
├── blur/
├── masks/                f_0001.png, f_0002.png, …   (255=유지, 0=제외)
└── mask_preview/         UI 검증용 빨강 오버레이
```

| 파이프라인 | 프레임 경로 | 마스크 경로 | 비고                                   |
|-----------|-------------|-------------|---------------------------------------|
| 3DGS      | `sharp/`    | `masks/`    | `<basename>.png`                      |
| 2DGS      | `sharp/`    | `masks/`    | `<basename>.png`                      |
| COLMAP    | `sharp/`    | `masks/`    | 필요 시 `<full_name>.png`로 리네임    |

---

## 주요 엔드포인트

| 메서드 | 경로                           | 용도                                     |
|--------|--------------------------------|------------------------------------------|
| GET    | `/health`                      | 버전 / 디바이스 / SAM2·DINO 상태         |
| POST   | `/upload`                      | 영상 업로드(Multipart)                   |
| GET    | `/metadata/{file_id}`          | FFprobe 메타데이터                       |
| WS     | `/ws/process/{file_id}`        | 프레임 추출 + 블러 점수 파이프라인       |
| POST   | `/change-model/{size}`         | SAM 2.1 사이즈 전환                      |
| POST   | `/segment`                     | Phase 1 · 포인트 단일 프레임             |
| WS     | `/ws/segment-video`            | Phase 2 · 동영상 전파                    |
| POST   | `/segment-text`                | Phase 4 · 텍스트 단일 프레임             |
| WS     | `/ws/segment-text-batch`       | Phase 4 · 텍스트 일괄                    |
| POST   | `/reconstruction-mask-info`    | Phase 3 · 마스크 경로·개수               |
| POST   | `/generate-mask-previews`      | 빨강 오버레이 그리드 생성                |
| GET    | `/overlay-file?path=…`         | 절대 경로 오버레이·마스크 파일 서빙      |
| POST   | `/reclassify`                  | 프레임을 `sharp/` ↔ `blur/`로 이동       |
| GET    | `/directories`                 | 출력 디렉터리 트리                       |
| POST   | `/create-directory`            | 서브 폴더 생성                           |
| POST   | `/sync-folders`                | 기존 결과에 임계값 재적용                |

---

## 프로젝트 구조

```
backend/
├── main.py             FastAPI 앱 · 모든 엔드포인트
├── ai_engine.py        SAMEngine 클래스 + 마스크 I/O 헬퍼
├── requirements.txt
├── output/             추출 프레임 (sharp/ + blur/) · masks/
└── masks/preview/      포인트 프롬프트 프리뷰 오버레이

frontend/
├── src/App.tsx         단일 페이지 대시보드
└── package.json
```

---

## 모델 노트

- SAM 2.1 사이즈: **Tiny**(최속) → **Small** → **Base+**(권장) → **Large**(최고 품질).
- 레거시 명 `vit_b` / `vit_l` / `vit_h`는 자동으로 `tiny` / `base_plus` / `large`
  로 매핑 — 기존 프론트엔드 호환성 유지.
- Grounding DINO는 첫 텍스트 프롬프트 호출 시 지연 로딩.
- PyTorch 2.6의 `weights_only` 동작을 우회하는 몽키패치 포함.

---

## 문제 해결

- **`No module named 'sam2'`** — 위의 git 설치 라인을 실행. PyPI에 미배포.
- **`invalid literal for int(): 'f_0033'`** — V4.0.0에서 해결. 비디오
  프레딕터가 임시 디렉터리에 정수명 심볼릭 링크를 생성한 뒤 호출하므로
  `f_0001.jpg` 같은 파일명도 그대로 동작.
- **`TypeError: … got an unexpected keyword argument 'box_threshold'`**  —
  `transformers ≥ 4.51`에서 인자명이 `threshold=`로 바뀐 이슈. 신·구 양쪽을
  시도하는 try/fallback 쉼으로 해결.
- **프리뷰 그리드가 비어있음** — `<scene>/masks/` 실제 파일 유무 확인.
  백엔드 로그에 `[PREVIEW] scene=… items=N` 출력. 씬이 `OUTPUT_DIR` 외부에
  있으면 `/images/…` 대신 `/overlay-file?path=…`로 서빙됨.
- **두 패스가 서로 덮어씀** — `[04]` 헤더의 COMBINE 체크박스를 켠 상태로 유지.

전체 변경 로그는 `HISTORY.md` 참조.
