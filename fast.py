# 1. Library imports marche stp
import uvicorn
from fastapi import FastAPI
from model import IrisModel, IrisSpecies

# 2. Create the app object
app = FastAPI()
model = IrisModel()

@app.get('/')
def home():
    return {'message':'Welcome'}

@app.get('/predict')
def predict_species(iris: IrisSpecies):
    data = iris.dict()
    prediction, probability = model.predict_species(
        data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']
    )
    return {
        'prediction': prediction,
        'probability': probability
    }


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)