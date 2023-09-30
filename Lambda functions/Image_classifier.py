import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer
import ast

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-09-18-17-55-18-474"

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['image_data'])
    
    # Instantiate a Predictor
    predictor = sagemaker.Predictor(endpoint_name=ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image).decode('utf-8')
    

    # We return the data back to the Step Function    
    event["inferences"] = ast.literal_eval(inferences)
    return {
        'statusCode': 200,
        "image_data":event["image_data"],
        "inferences":event["inferences"]
    }
    