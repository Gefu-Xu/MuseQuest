# --------------- Functions for Making OpenAI API Calls ---------------
# 1) Calculate the cost of API calls based on the number of tokens used
def calculate_token_cost(model, input_token_count=0, output_token_count=0):
    input_token_price = 0
    output_token_price = 0
    # define token prices (dollars per million tokens, price as of Jan 2025) based on the model
    if model == 'gpt-4o':
        input_token_price = 2.5
        output_token_price = 10.0
    elif model == 'gpt-4o-mini':
        input_token_price = 0.15
        output_token_price = 0.6
    # calculate the total cost of API calls
    cost = (input_token_count / 1_000_000 * input_token_price +
            output_token_count / 1_000_000 * output_token_price)
    return round(cost, 2)

# 2) Encode an image to base64 format for OpenAI multi-modal model processing
import base64
import io
def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")  # save the image to an in-memory BytesIO buffer
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# 3) Send a user query with an image using the OpenAI API 
import requests
import time
def send_user_query_with_image(img_path, user_question, max_retry=5):
    base64_image = encode_image(img_path)
    api_key = 'your OpenAI API key' # replace with your OpenAI API key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o",    #latest multi-modal model
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{user_question}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }
    # attempt API call with exponential backoff for throttling handling
    for attempt in range(1, max_retry + 1):
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 429: #handle rate limiting with exponential backoff
            wait = 2 ** attempt
            print(f"Retrying in {wait} seconds...", end='')
            time.sleep(wait)
        else:                           #for non-throttling status, go to next step
            break
    if response.status_code == 200:     #successful API response
        response_content = response.json()["choices"][0]["message"]["content"]
        input_tokens = response.json()["usage"]["prompt_tokens"]
        output_tokens = response.json()["usage"]["completion_tokens"]
        return response_content, input_tokens, output_tokens
    else:                               #handle API failures
        print(f'\nAPI call failure: {response.status_code}')
        print(f'API call reponse: {response.json()}')
        return None, 0, 0

# 4) Perform inference on the test dataset using OpenAI API, save image thumbnails, and save inference results to a CSV file
# Note: The saved image thumbnails and the CSV file will be used in plot_similarity_scores() next step
import os
import pandas as pd
import json
import my_utils
def inference_openai(model, processor, dataset, report_csv):
    print(f'==> Performing inference and saving results to {report_csv} ...')

    # Prepare directories for image thumbnails and CSV file
    report_csv_dir = os.path.dirname(report_csv)
    thumbnail_dir = os.path.join(report_csv_dir, 'thumbnail')
    if not os.path.exists(report_csv_dir):
        os.makedirs(report_csv_dir)
    if not os.path.exists(thumbnail_dir):
        os.makedirs(thumbnail_dir)

    report_df = pd.DataFrame(columns=['data_id', 'item_name', 'image_name', 'user_question', 'model_generated_answer', 'ground_truth_answer', 'semantic_similarity_score'])
    accumulated_semantic_similarity_score = 0
    total_input_tokens = 0
    total_output_tokens = 0
    for idx, data in enumerate(dataset):
        print(f'==> [Processing data: {idx + 1}/{len(dataset)}]  data_id: {data["data_id"]}, item_name: {data["item_name"]}, image_name: {data["image_name"]}')
        # Save a thumbnail of the current image
        image = data['image']
        image_resized = data['image'].resize((100, 100))
        thumbnail = data['image'].resize((20, 20))
        thumbnail.save(os.path.join(thumbnail_dir, data["image_name"]))
        # Extract the user question and ground truth answer from the current dialog
        dialog_json = json.loads(data['dialog'])
        user_msg = dialog_json['messages'][0] 
        assistant_msg = dialog_json['messages'][1]
        user_question = user_msg['content'][1]['text']
        ground_truth_answer = assistant_msg['content'][0]['text']
        # Send the user query along with the image to OpenAI API and get the generated response
        generated_answer, input_tokens, output_tokens = send_user_query_with_image(image, user_question)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cost = calculate_token_cost('gpt-4o', total_input_tokens, total_output_tokens)
        # Display the response generated by OpenAI API and evaluate its quality using semantic similarity scores
        semantic_similarity_score = my_utils.calculate_semantic_similarity_score(model, processor, generated_answer, ground_truth_answer)
        image_resized.show()
        print(f'==> Original user question: {user_question}')
        print(f'==> OpenAI GPT-4o generated answer: {generated_answer}')
        print(f'==> Ground truth answer: {ground_truth_answer}')
        print(f'==> Semantic similarity score: {semantic_similarity_score}')
        print(f'==> OpenAI GPT-4o total_tokens: {total_input_tokens}(input), {total_output_tokens}(output), total_cost: ${total_cost}')
        # Save results in the CSV file
        data_df = pd.DataFrame([{'data_id': data['data_id'],
                                    'item_name': data['item_name'],
                                    'image_name': data['image_name'],
                                    'user_question': user_question,
                                    'model_generated_answer': generated_answer,
                                    'ground_truth_answer': ground_truth_answer,
                                    'semantic_similarity_score': semantic_similarity_score.item()}])
        report_df = pd.concat([report_df, data_df], ignore_index=True)
        accumulated_semantic_similarity_score += semantic_similarity_score.item()
    # Calculate and save average similarity score for the dataset
    ave_semantic_similarity_score = round(accumulated_semantic_similarity_score / len(dataset), 3)
    print(f'==> Average semantic similarity score: {ave_semantic_similarity_score} (0 is the best, 2 is the worst)')
    report_df = pd.concat([report_df, pd.DataFrame([{'item_name': 'average_semantic_similarity_score', 'semantic_similarity_score': ave_semantic_similarity_score}])], ignore_index=True)
    report_df.to_csv(report_csv, index=False, encoding='utf-8')
