import os
import torch
import gradio as gr
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def load_models():
    # build model and tokenizer
    model_name_dict = {#'nllb-distilled-600M': 'facebook/nllb-200-distilled-600M',
                  #'nllb-1.3B': 'facebook/nllb-200-1.3B',
                  'nllb-distilled-1.3B': 'facebook/nllb-200-distilled-1.3B',
                  #'nllb-3.3B': 'facebook/nllb-200-3.3B',
                  }

    model_dict = {}

    for call_name, real_name in model_name_dict.items():
        print('\tLoading model: %s' % call_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(real_name)
        tokenizer = AutoTokenizer.from_pretrained(real_name)
        model_dict[call_name+'_model'] = model
        model_dict[call_name+'_tokenizer'] = tokenizer

    return model_dict


def translation(text):
    if len(model_dict) == 2:
        model_name = 'nllb-distilled-1.3B'

    start_time = time.time()
    source = "eng_Latn"
    target = "tha_Thai"

    model = model_dict[model_name + '_model']
    tokenizer = model_dict[model_name + '_tokenizer']

    translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source, tgt_lang=target ,device="cuda")
    output = translator(text, max_length=1000)

    end_time = time.time()

    output = output[0]['translation_text']
    # result = {'inference_time': end_time - start_time,
    #           'source': source,
    #           'target': target,
    #           'result': output}
    return output


if __name__ == '__main__':
    print('\tinit models')

    global model_dict

    model_dict = load_models()
    list_text =["for text in list_text:",
                "In the scale of optimal virulence, vertical transmission tends to progress benign symbiosis, so is a critical idea for evolutionary medicine. Because the ability of reproducibility of pathogen in the host is the leading cause of pathogen to pass from mother to child, Its transmissibility tends to be inversely related to their virulence. Although HIV is transmitted through perinatal transmission, it is vertical transmission is not the primary mode of transmission. in addition to the new medicine decreased the frequency of vertical transmission of HIV. The incidence of perinatal HIV cases in the United States has decreased as a result of the implementation of recommendations on HIV counselling and voluntary testing practices and the use of zidovudine therapy to reduce perinatal HIV transmission. In dual inheritance theory, vertical transmission refers to the passing of cultural traits from parents to children."]
    for text in list_text:
        print(translation(text))
    
    
