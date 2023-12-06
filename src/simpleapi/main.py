import shutil
from tempfile import NamedTemporaryFile
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from pydantic import BaseModel
import cv2
import numpy as np


class Vertex(BaseModel):
    x: int
    y: int


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/hello/{name}")
def hello_name(name: str):
    return {"Hello": name}


@app.post("/upload-image/")
def upload_image(image: UploadFile):
    try:
        file_path = f"upload_images/{image.filename}"

        with open(file_path, "wb+") as buffer:
            shutil.copyfileobj(image.file, buffer)
        return {"message": f"File '{image.filename}' uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/inpaint-image/",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def inpaint_image(
    file: UploadFile = File(...),
    rectangles: List[List[Vertex]] = None,
) -> Response:
    """Upload Image and inpaint it.

    Args:
        file (UploadFile, optional): _description_. Defaults to File(...).
        vertices (List[Vertex], optional): _description_. Defaults to None.
    """
    # テスト用データ
    # rectangles = [
    #     [
    #         Vertex(x=150, y=150),
    #         Vertex(x=150, y=100),
    #         Vertex(x=100, y=100),
    #         Vertex(x=100, y=150),
    #     ]
    # ]
    try:
        # 画像を一時保存
        with NamedTemporaryFile(delete=False) as buffer:
            shutil.copyfileobj(file.file, buffer)
            file_path = buffer.name
        # 画像の読み込み
        image = cv2.imread(file_path)

        # マスク画像の作成
        mask = np.zeros(image.shape[:2], np.uint8)
        for vertices in rectangles:
            rect = np.array([[v.x, v.y] for v in vertices], np.int32)
            # rect = rect.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [rect], 255)

        # inpaint
        dst = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

        # 画像の保存
        cv2.imwrite("input.png", image)
        cv2.imwrite("output.png", dst)
        cv2.imwrite("mask.png", mask)

        # 画像の返却
        # 処理された画像の保存または直接レスポンスとして返却
        _, encoded_image = cv2.imencode(".png", dst)
        return Response(encoded_image.tobytes(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
