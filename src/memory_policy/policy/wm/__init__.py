# instruction_memory.py
import torch
import numpy as np
import torch.nn.functional as F
from memory_policy.modules import SigLIP2
from typing import Dict, Tuple, Optional

# 이미 정의해 둔 SigLIP2 클래스를 사용합니다.

class InstructionMemory:
    """
    forward(images, instruction, resnet_patches) 호출 시
    - SigLIP2로 이미지 임베딩과 텍스트 임베딩 계산
    - 텍스트와 가장 유사한 이미지 하나를 선택
    - {instruction: {"siglip_image": ..., "siglip_text": ..., "resnet_patch": ..., "score": ...}} 형태로 저장
    """

    def __init__(self, siglip: "SigLIP2", store_on_cpu: bool = True):
        self.siglip = siglip
        self.store_on_cpu = store_on_cpu
        # 🔹 텍스트 임베딩 캐시 추가
        self.text_cache: Dict[str, torch.Tensor] = {}
        # 🔹 최고점만 저장하는 베스트 뱅크
        self.best_bank: Dict[str, Dict[str, torch.Tensor]] = {}

        self.text_cache: Dict[str, torch.Tensor] = {}
        self.best_bank: Dict[str, Dict[str, torch.Tensor]] = {}

    @torch.no_grad()
    def ensure_text_embedding(self, instruction: str) -> torch.Tensor:
        """
        instruction 텍스트 임베딩을 캐시하고 반환한다.
        """
        if instruction not in self.text_cache:
            txt_emb = self.siglip.embed_texts(instruction)  # (1, D)
            if self.store_on_cpu:
                txt_emb = txt_emb.cpu()
            self.text_cache[instruction] = txt_emb[0]  # (D,)
        return self.text_cache[instruction]

    @torch.no_grad()
    def update(
        self,
        image_hwc: np.ndarray,
        instruction: str,
        resnet_patch: torch.Tensor,
        bgr: bool = False,
        index_hint: Optional[int] = None,
    ) -> Tuple[float, bool]:
        """
        스트리밍 이미지 한 장을 받아 베스트 갱신을 시도한다.

        image_hwc   HWC uint8 numpy
        instruction 문자열
        resnet_patch (C,H,W) 또는 (1,C,H,W) 텐서
        bgr         OpenCV BGR 입력이면 True
        index_hint  외부에서 이 프레임의 인덱스를 관리하고 있으면 넘길 수 있음

        return
          score        이번 프레임의 cosine similarity
          is_best      이번 프레임이 현재까지 최고였는지 여부
        """
        # 1 텍스트 임베딩 확보
        txt = self.ensure_text_embedding(instruction)    # (D,)
        txt_dev = txt.device

        # 2 이미지 임베딩  배치 1로 계산
        img_emb = self.siglip.embed_images(image_hwc, bgr=bgr)  # (1, D)
        if self.store_on_cpu:
            img_emb = img_emb.cpu()

        # 3 similarity
        # txt shape (D,) → (1,D)로 맞춰 matmul
        score = float((txt.unsqueeze(0) @ img_emb.T).item())

        # 4 resnet_patch 정리
        if not isinstance(resnet_patch, torch.Tensor):
            raise TypeError("resnet_patch는 torch.Tensor여야 합니다")
        if resnet_patch.dim() == 4:
            if resnet_patch.size(0) != 1:
                raise ValueError("resnet_patch가 배치라면 배치 크기는 1이어야 합니다")
            resnet_patch = resnet_patch[0]  # (C,H,W)
        elif resnet_patch.dim() != 3:
            raise ValueError("resnet_patch는 (C,H,W) 또는 (1,C,H,W) 여야 합니다")

        if self.store_on_cpu:
            resnet_patch = resnet_patch.detach().cpu()

        # 5 베스트 갱신
        entry = self.best_bank.get(instruction)
        frame_idx = index_hint if index_hint is not None else int(entry["count"]) + 1 if entry else 0
        is_best = False
        if entry is None or score > float(entry["score"]):
            self.best_bank[instruction] = {
                "siglip_image": img_emb[0].clone(),
                "siglip_text": txt.clone(),
                "resnet_patch": resnet_patch.clone(),
                "score": torch.tensor(score),
                "index": torch.tensor(frame_idx),
                "count": torch.tensor((0 if entry is None else int(entry["count"])) + 1),
            }
            is_best = True
        else:
            # 베스트 유지하면서 count만 증가
            entry["count"] = torch.tensor(int(entry["count"]) + 1)

        return score, is_best

    def get_best(self, instruction: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        해당 instruction의 현재까지 베스트 결과를 반환한다.
        반환 dict 키
          siglip_image  (D,)
          siglip_text   (D,)
          resnet_patch  (C,H,W)
          score         float tensor
          index         int tensor  업데이트 당시 프레임 인덱스 힌트
          count         int tensor  지금까지 처리한 프레임 수
        """
        return self.best_bank.get(instruction)

    def clear_instruction(self, instruction: str):
        """
        특정 instruction의 캐시와 베스트를 초기화한다.
        """
        if instruction in self.text_cache:
            del self.text_cache[instruction]
        if instruction in self.best_bank:
            del self.best_bank[instruction]

    def clear_all(self):
        """
        모든 캐시와 결과를 초기화한다.
        """
        self.text_cache.clear()
        self.best_bank.clear()

    def state_dict(self):
        return {
            "store_on_cpu": self.store_on_cpu,
            "bank": {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in self.best_bank.items()}
        }

    def load_state_dict(self, state):
        self.store_on_cpu = bool(state.get("store_on_cpu", True))
        bank = state.get("bank", {})
        self.best_bank = {
            k: {kk: (vv.clone() if isinstance(vv, torch.Tensor) else vv) for kk, vv in v.items()} for k, v in bank.items()}


    """from here, memory io"""

    def _stack_bank(self):
        """
        self.best_bank에서
        texts: [str]
        txts:  (T, D)
        imgs:  (T, D)
        patches: (T, C, H, W)
        를 만든다
        """
        if not self.best_bank:
            raise RuntimeError("best_bank이 비어 있음")

        texts = list(self.best_bank.keys())
        txts = torch.stack([self.best_bank[t]["siglip_text"] for t in texts], dim=0)
        imgs = torch.stack([self.best_bank[t]["siglip_image"] for t in texts], dim=0)
        patches = torch.stack([self.best_bank[t]["resnet_patch"] for t in texts], dim=0)
        return texts, txts, imgs, patches

    @torch.no_grad()
    def most_similar_text_tuple(self, instruction: str):
        """
        현재 instruction과 텍스트 임베딩 기준으로 가장 유사한 과거 항목을 반환
        return
        best_text, best_score, best_entry(dict)
        """
        query = self.ensure_text_embedding(instruction)           # (D,)
        texts, txts, imgs, patches = self._stack_bank()           # T, (T,D), (T,D), (T,C,H,W)
        query = torch.nn.functional.normalize(query, dim=0)
        txts = torch.nn.functional.normalize(txts, dim=1)

        sims = txts @ query                                      # (T,)
        idx = int(torch.argmax(sims).item())
        best_text = texts[idx]
        best_score = float(sims[idx])
        best_entry = self.best_bank[best_text]
        return best_text, best_score, best_entry

    @torch.no_grad()
    def combine_patches_by_img_similarity(
        self,
        instruction: str,
        topk: int | None = None,
        temperature: float = 0.05,
        return_weights: bool = False,
    ):
        """
        현재 텍스트 임베딩과 과거 이미지 임베딩의 유사도로 패치를 선형 결합
        return
        mixed_patch (C,H,W)
        (선택) indices(T_sel,), weights(T_sel,)
        """
        # 1 텍스트 임베딩과 메모리 행렬
        query_txt = self.ensure_text_embedding(instruction)      # (D,)
        texts, txts, imgs, patches = self._stack_bank()          # imgs: (T,D), patches: (T,C,H,W)

        # 2 점수 계산  txt vs img
        q = torch.nn.functional.normalize(query_txt, dim=0)      # (D,)
        K = torch.nn.functional.normalize(imgs, dim=1)           # (T,D)
        sims = K @ q                                             # (T,)

        # 3 선택 전략  상위 k만 쓰거나 전체 사용
        if topk is not None and topk < sims.numel():
            vals, idxs = torch.topk(sims, k=topk, dim=0)
            sims_sel = vals
            patches_sel = patches[idxs]                          # (k,C,H,W)
            idxs_sel = idxs
        else:
            sims_sel = sims
            patches_sel = patches                                # (T,C,H,W)
            idxs_sel = torch.arange(sims.shape[0])

        # 4 온도 소프트맥스 가중치
        w = torch.softmax(sims_sel / max(1e-6, temperature), dim=0)   # (T_sel,)

        # 5 선형 결합  Σ w_i * patch_i
        mixed = torch.tensordot(w, patches_sel, dims=([0],[0]))       # (C,H,W)

        if return_weights:
            return mixed, idxs_sel, w
        return mixed





if __name__ == "__main__":

    sig = SigLIP2(model_id="google/siglip2-base-patch16-224")
    mem = StreamingInstructionMemory(sig)

    instruction = "pick up the red cube"

    # 스트리밍 루프 가정
    for t in range(10):
        img = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
        res_patch = torch.randn(512, 16, 16)
        score, is_best = mem.update(img, instruction, res_patch, bgr=False, index_hint=t)
        if is_best:
            print(f"[t={t}] new best score {score:.4f}")

    best = mem.get_best(instruction)
    print("final best score", float(best["score"]))
    print("best index", int(best["index"]))
    print("siglip image emb shape", tuple(best["siglip_image"].shape))
    print("resnet patch shape", tuple(best["resnet_patch"].shape))


    # 현재 텍스트와 가장 유사한 과거 텍스트 찾기
    best_text, best_score, best_entry = mem.most_similar_text_tuple("insert the peg")

    # 현재 텍스트 기준으로 과거 키프레임 패치들을 이미지 유사도로 혼합
    mixed_patch, idxs, weights = mem.combine_patches_by_img_similarity(
        "insert the peg", topk=4, temperature=0.07, return_weights=True
    )


    """
    1. state_dict
        현재 객체의 핵심 상태를 딕셔너리로 반환
        포함 내용
        store_on_cpu 설정값
        best_bank의 깊은 복사본
        vv.clone()을 써서 텐서가 원본과 메모리를 공유하지 않도록 안전하게 복사
        텐서가 아닌 값은 그대로 넣음
    2. load_state_dict
        state_dict로부터 객체 상태를 복원
        포함 내용
        store_on_cpu 값을 복원
        bank에 들어 있는 항목으로 best_bank 재구성
        텐서는 clone()으로 새 텐서로 만들어 참조 관계를 끊음
        텐서가 아닌 값은 그대로 복사
    3. 왜 clone을 쓰나
        저장된 텐서를 그대로 참조하면 이후 수정 시 원본과 저장본이 서로 영향을 주는 문제가 생길 수 있어요
        clone()으로 분리해 부작용을 방지합니다

    # 저장
    state = mem.state_dict()
    torch.save(state, "mem_state.pt")

    # 로드
    state = torch.load("mem_state.pt", map_location="cpu")
    mem.load_state_dict(state)
    """