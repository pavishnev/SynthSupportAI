from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from utils.ai_testing_utils import json_to_question_answer

model_name = "../all-MiniLM-L6-v2"

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_closest_sentence(source_sentence, sentences):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    print("Sentence embeddings:")
    print(sentence_embeddings)

    # Tokenize and compute embedding for the input sentence
    input_encoded = tokenizer(source_sentence, return_tensors='pt')
    with torch.no_grad():
        input_embedding = mean_pooling(model(**input_encoded), input_encoded['attention_mask'])
        input_embedding = F.normalize(input_embedding, p=2, dim=1)

    # Calculate cosine similarity between input embedding and each sentence embedding
    cosine_similarities = F.cosine_similarity(input_embedding, sentence_embeddings)
    
    # Find the index of the sentence with the highest similarity
    closest_sentence_index = torch.argmax(cosine_similarities).item()

    # Get the closest sentence
    closest_sentence = sentences[closest_sentence_index]

    print("Input sentence:", source_sentence)
    print("Closest sentence:", closest_sentence)
    print("Cosine similarity:", cosine_similarities[closest_sentence_index].item())
    return closest_sentence, closest_sentence_index, cosine_similarities[closest_sentence_index].item()

def Generate(question, json_file_path):
    data = json_to_question_answer(json_file_path)
    closest_question, closest_question_index, probability=get_closest_sentence(question,data['question'])
    return {
            "answer":data['answer'][closest_question_index],
            "score":probability
            }

#Generate("I am unable to access my Current Account", "docs/bank_faqs.json")
