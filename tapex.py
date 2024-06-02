from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd
from utils.ai_testing_utils import json_to_question_answer

def Generate(question, json_file_path):
    model_name = "../tapex-base-finetuned-wikisql"
    tokenizer = TapexTokenizer.from_pretrained(model_name, 
                                               add_prefix_space=True,
                                               truncation=True,
                                               max_length=512)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    data = json_to_question_answer(json_file_path)
    table = pd.DataFrame.from_dict(data)

    # tapex accepts uncased input since it is pre-trained on the uncased corpus
    encoding = tokenizer(table=table, query=question, return_tensors="pt")
    aaa = encoding['input_ids'].size()[1]
    outputs = model.generate(**encoding)

    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

Generate("I am unable to access my Current Account", "docs/bank_faqs.json")