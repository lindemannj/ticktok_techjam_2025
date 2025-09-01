import base64
import traceback
import services.config as config

from services.image_services import process_image_bytes_and_pixelate
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()


@app.get("/")
def health_check():
    """
    This endpoint performs a simple health check.

    It returns a JSON object with a status message and a status code
    to confirm that the API Gateway server is running and accessible.
    """
    return {"status": "api-gateway server is running", "status-code": 200}


@app.post("/process-image/")
async def process_image_endpoint(file: UploadFile = File(...)):
    """
    This endpoint handles the processing of an uploaded image.

    It receives an image file, reads its binary data, and sends it to the
    `process_image_bytes_and_pixelate` service for processing. The service
    pixelates any identified text and faces. The resulting image is then
    encoded to a Base64 string and returned along with processing metadata.

    The endpoint includes a robust `try-except` block to catch and handle any
    processing errors, returning a 500 status code with an appropriate error
    message if an issue occurs.
    """
    try:
        image_bytes = await file.read()

        processed_jpeg_bytes, metadata = process_image_bytes_and_pixelate(
            image_bytes,
            min_text_conf=config.MIN_TEXT_CONF,
            text_pixel_size=config.TEXT_PIXEL_SIZE,
            face_pixel_size=config.FACE_PIXEL_SIZE,
            tesseract_lang=config.TESSERACT_LANG,
            tesseract_psm=config.TESSERACT_PSM,
        )

        encoded_image_string = base64.b64encode(processed_jpeg_bytes).decode("utf-8")
        return JSONResponse(content={
            "processed_image_string": encoded_image_string,
            "metadata": metadata
        })

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Ein Fehler ist aufgetreten: {e}\n{tb}")
        return JSONResponse(status_code=500, content={"error": f"Verarbeitung fehlgeschlagen: {e}"})
