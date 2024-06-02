from utils.ai_testing_utils import measure_context_generator
from roberta_base import Generate as roberta
from distilled import Generate as distilled
from mini_lm import Generate as mini_lm


from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

questions=[
    "On the magic chessboard, what piece did Hermione play with?",
    "What did first-year students have to do during the practical part of the Transfiguration exam?",
    "What was the first password to the Gryffindor common room?",
    "What breed of dragon was Norbert, Hagrid's little dragon?",
    "Who gave Harry Potter the Nimbus 2000 broom?"
]

exp_ans=["bishop", "turn a mouse into a snuff box", "Caput Draconis", "Norwegian Ridgeback", "Minerva McGonagall"]
2
questions1=[
    "When The Knockout was released?",
    "Who was the director of Those Love Pangs?",
    "Who casted in The Adventures of the American Rabbit?",
    "What genre is Manhattan Madness?",
    "Who casted in Hulda from Holland?"
]
exp_ans1=["1914", 
          "Charlie Chaplin", 
          "Barry Gordon", 
          "comedy", 
          "Mary Pickford, Frank Losee"]

custom_parameters = {
    "do_sample": False,
    "temperature": 1,
    "top_p": 1.0,
    "top_k": 20,
    "repetition_penalty": 2.0}
methods = [roberta, distilled]

#measure_context_generator(roberta,custom_parameters, questions, exp_ans, "docs/harry_potter_full.txt")

#print(roberta("Tell me about Discord", 
#              "docs/discord.txt", 3
#              custom_parameters))

def method_one():
    banking_questions=['I need to open a Institutional account',
                       'Tell me about Smartup Solutions',
                       'I need business benefits of Payzapp',
                       'Issuing a certificate of TDS ',
                       'Can I open an RFC in euro currency?'
                       ]
    bank_exp_answers=['Non-individual (Current account) AOD to be completed Registration proof with any state regulatory...',
                      'A SMART solution for your StartUp. SmartUp is here to assist you in achieving your startup goals with smart financial tools, smart advisory services and technology.',
                      'Payzapp for business is arevolutionary invoicing solution by HDFC Bank which enables start-ups...',
                      'You will receive a consolidated TDS Certificate in Form 16A, for TDS deducted during a financial year...',
                      'An RFC Domestic Account can be opened in three different currencies, i.e. USD / Euro / GBP.'
                      ]
    
    measure_context_generator(mini_lm, 
                              questions=banking_questions,
                              expected_answers=bank_exp_answers,
                              context_file_path="docs/bank_faqs.json"
                              )
    print("Вы выбрали метод тестирования banking faq")
    
    # Ваш код для метода 1

def method_two():
    print("Вы выбрали метод 2.")
    
    measure_context_generator(roberta,
                              questions1, 
                              exp_ans1, 
                              "docs/movies_plotless.txt")

def methor_three():
    print("Вы выбрали метод 3.")
    
    measure_context_generator(distilled,
                              questions1, 
                              exp_ans1, 
                              "docs/movies_plotless_shorten.txt")

def main():
    while True:
        print("\nВыбериите метод:")
        print("0 - Выйти из программы")
        print("1 - Тест для faq")
        print("2 - Roberta (Harry potter)")
        print("3 - Distilled (Harry potter)")

        choice = input("Ваш выбор: ")

        if choice == "0":
            print("Программа завершена.")
            break
        elif choice == "1":
            method_one()
        elif choice == "2":
            method_two()
        elif choice == "3":
            methor_three()
        else:
            print("Неверный ввод. Пожалуйста, введите 0, 1 или 2, 3.")

if __name__ == "__main__":
    main()
