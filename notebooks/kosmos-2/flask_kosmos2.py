# -*- coding: utf-8 -*-

import json
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from flask import Flask, request, jsonify

app = Flask(__name__)

MP = "J:/model/mllm-model/ydshieh-kosmos-2-patch14-224"
model = AutoModelForVision2Seq.from_pretrained(MP, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MP, trust_remote_code=True)

def entities_to_json(entities):
    result = []
    for e in entities:
        label = e[0]
        box_coords = e[1]
        box_size = e[2][0]
        entity_result = {
            "label": label,
            "boundingBoxPosition": {"x": box_coords[0], "y": box_coords[1]},
            "boundingBox": {"x_min": box_size[0], "y_min": box_size[1], "x_max": box_size[2], "y_max": box_size[3]}
        }
        print(entity_result)
        result.append(entity_result)

    return result

@app.route("/process_prompt", methods=["POST"])
def process_prompt():
    try:
        # Get the uploaded image data from the POST request
        uploaded_file = request.files['image']
        prompt = request.form.get("image")
        image = Image.open(uploaded_file.stream)
        print(image.size)

        inputs = processor(text=prompt,
                           images=image,
                           return_tensors="pt")
        generated_ids = model.generate(
            pixel_values = inputs["pixel_values"],
            input_ids = inputs["input_ids"][:, :-1],
            img_features = None,
            img_attn_mask = inputs["attention_mask"][:, :-1],
            use_cache = True,
            max_new_tokens = 64,
        )
        generated_text = processor.batch_deocde(generated_ids, skip_special_tokens=True)[0]
        # By default, the generated  text is cleanup and the entities are extracted.
        processed_text, entities = processor.post_process_generation(generated_text)
        parsed_entities = entities_to_json(entities)
        print(generated_text)
        print(processed_text)
        return jsonify({"message": processed_text, 'entities': parsed_entities})

    except Exception as e:
            print(e)

if __name__ == '__main__':
    app.run(host='localhost', port=8005)