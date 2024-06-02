from transformers import RobertaConfig
import time
import os
import datetime
import json

def measure_context_generator(generator, questions, expected_answers, context_file_path):
    directory_path = "logs/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file_path = os.path.join(directory_path, f"log_{current_time}_ssai.txt")
    #with open(log_file_path, 'w') as log_file:
     #   log_file.write(str(custom_params))
    with open(log_file_path, 'w') as log_file:
        for i, q in enumerate(questions):
            start_time = time.time()
            output = generator(q, context_file_path)
            end_time = time.time()
            exec_time = end_time-start_time
            log = f"|||||| \nMethod: {generator.__name__} \nQuestion: {q} \nAnswer: {output[0]['answer']} \nExp_answer: {expected_answers[i]} \nTime: {exec_time} \nScore: {output[0]['score']}\n"
            print(log)
            log_file.write(log)
    
def json_to_question_answer(json_file_path, limit=300):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    data = {
    "question": [],
    "answer": []
    }
     
    for i, question_set in enumerate(json_data["bank"]["accounts"]):
        if i>=limit:
            break
        data["question"].append(question_set[0])
        data["answer"].append(question_set[1])
    
    return data
    