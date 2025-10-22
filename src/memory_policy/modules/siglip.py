# siglip2_similarity.py
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel


class SigLIP2:
    """
    SigLIP2 이미지/텍스트 임베딩과 유사도 계산을 한 클래스에 묶은 유틸
    """
    def __init__(self, model_id: str = "google/siglip2-base-patch16-224", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    @staticmethod
    def _ensure_list_of_pils(images, bgr: bool):
        """
        images
          np.ndarray HWC uint8
          np.ndarray BHWC uint8
          list[np.ndarray HWC uint8]
        return
          list[PIL.Image]
        """
        def to_pil(arr: np.ndarray) -> Image.Image:
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if bgr:
                arr = arr[..., ::-1]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")

        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                return [to_pil(images)]
            if images.ndim == 4:
                return [to_pil(img) for img in images]
            raise ValueError("numpy 이미지는 HWC 또는 BHWC 형식이어야 합니다")
        if isinstance(images, list):
            return [to_pil(img) for img in images]
        raise TypeError("images는 numpy 배열 또는 numpy 배열 리스트여야 합니다")

    @torch.no_grad()
    def embed_images(self, images, bgr: bool = False) -> torch.Tensor:
        """
        images에 numpy 이미지를 넣는다
        bgr가 True이면 OpenCV BGR을 RGB로 변환한다
        return
          (B, D) 정규화된 텐서
        """
        pil_images = self._ensure_list_of_pils(images, bgr=bgr)
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def embed_texts(self, texts) -> torch.Tensor:
        """
        texts 문자열 또는 문자열 리스트
        return
          (T, D) 정규화된 텐서
        """
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def similarity(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        image_embeds (B, D)
        text_embeds  (T, D)
        return       (T, B) cosine similarity
        """
        return text_embeds @ image_embeds.T

    @torch.no_grad()
    def most_similar_image_for_text(self, images, text: str, bgr: bool = False):
        """
        여러 numpy 이미지 중 한 문장과 가장 유사한 이미지를 찾는다
        return
          sims (B,)
          best_idx (int)
        """
        img_emb = self.embed_images(images, bgr=bgr)
        txt_emb = self.embed_texts(text)
        sims = self.similarity(img_emb, txt_emb).squeeze(0)
        best_idx = int(torch.argmax(sims).item())
        return sims, best_idx

    @torch.no_grad()
    def most_similar_image_for_each_text(self, images, texts, bgr: bool = False):
        """
        여러 텍스트 각각에 대해 가장 유사한 이미지를 찾는다
        return
          sims (T, B)
          best_idx_per_text (T,)
        """
        img_emb = self.embed_images(images, bgr=bgr)
        txt_emb = self.embed_texts(texts)
        sims = self.similarity(img_emb, txt_emb)
        best_idx = torch.argmax(sims, dim=1)
        return sims, best_idx


if __name__ == "__main__":
    sig = SigLIP2(model_id="google/siglip2-base-patch16-224")

    # 예시 단일 텍스트 vs 여러 numpy 이미지
    # imgs_bhwc는 예를 들어 shape (3, 256, 256, 3) uint8 라고 가정한다
    imgs_bhwc = np.random.randint(0, 255, size=(3, 256, 256, 3), dtype=np.uint8)
    sims, idx = sig.most_similar_image_for_text(imgs_bhwc, "a dog on the grass")
    print("sims:", [round(float(s), 4) for s in sims])
    print("best index:", idx)