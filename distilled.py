from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from utils.files_utils import read_file_content
from transformers import pipeline
custom_parameters = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50, 
    "repetition_penalty": 2.0,
    "max_new_tokens": 100}

def Generate(quest, context_file_path, custom_params= custom_parameters):
    model_name = '../distilbert-base-cased-distilled-squad'
    question_answerer = pipeline("question-answering", model=model_name)
    
    result = question_answerer(question=quest,
                                context=read_file_content(context_file_path),
                                **custom_params)

    return result

def Generate55(question, context_file_path, custom_params = custom_parameters):
    model_name = '../distilbert-base-cased-distilled-squad'
    
    # b) Load model & tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # a) Get predictions
    nlp = pipeline('question-answering', 
                   model=model, 
                   tokenizer=tokenizer,
                   device=0)
    
    QA_input = {
        'question': f'{question}',
        'context': f'{read_file_content(context_file_path)}'
        }
    output = nlp(QA_input,
                top_k=custom_params.get("top_k",10),
                top_p=custom_params.get("top_p",0.9),
                device='gpu')
    
    return output[0]