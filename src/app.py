import json 
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import numpy as np
from configs.config import MODEL_NAME, LABELS as labels, configs
from models.grading_model import ModelSkeleton
import tensorflow as tf
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles


model = tf.keras.models.load_model(MODEL_NAME)

def predict_similarity(sentence_a, sentence_b):
    sentence_pairs = np.array([[str(sentence_a), str(sentence_b)]])
    input_data = ModelSkeleton(
        sentence_pairs, labels=None, batch_size=12, shuffle=False, include_targets=False,
    )
    similarity_scores = model.predict(input_data[0])[0]
    
    label_scores = {label: float(score) for label, score in zip(labels, similarity_scores)}
    return label_scores



app = FastAPI(
    title="Quizzix AutoGrading_API",
    description="The AutoGrading API is a powerful tool that provides a simplified interface for integrating automated essay grading capabilities into educational applications, leveraging state-of-the-art NLP models such as BERT for accurate and efficient evaluation of student essays.",
    version="0.1.0", docs_url=None, redoc_url=None
)

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

class GradingResponse(BaseModel):
    grade_percentage: float 
    predicted_label: str 

class GradingRequest(BaseModel):
    student_answer: str 
    correct_answer: str 

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="../static/swagger-ui-bundle.js",
        swagger_css_url="../static/swagger-ui.css",
    )

@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get(path="/")
def read_root():
    return json.dumps({"message": "Hi"})

@app.post(path="/grade", response_model=GradingResponse)
def grade(request: GradingRequest) -> GradingResponse:
    student_answer = request.student_answer
    correct_answer = request.correct_answer
    predicted_label = predict_similarity(student_answer, correct_answer)
    print(predicted_label)
    return GradingResponse(grade_percentage=round(predicted_label['entailment'], 4), predicted_label='entailment')

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(app, host=configs['host'], port=configs['port'])
