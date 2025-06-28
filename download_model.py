from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_local_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # removed low_cpu_mem_usage
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)
