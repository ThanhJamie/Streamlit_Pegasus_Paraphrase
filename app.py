import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from sentence_splitter import SentenceSplitter
import evaluate
from parrot import Parrot
from sentence_splitter import SentenceSplitter
import pandas as pd


st.set_page_config(
    page_title="Pegasus - Parapraser",
    page_icon="🎈",
)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load metric form HuggingFace {BLUESCORE, TERSORE, ROUGE, METEOR, SACREBLEU}
# 'bleu': 1.0, 'precisions': [1.0, 1.0, 1.0, 1.0], 'brevity_penalty': 1.0, 'length_ratio': 1.1666666666666667, 'translation_length': 7, 'reference_length': 6}
bleu = evaluate.load('bleu')
# {'score': ter_score, 'num_edits': num_edits, 'ref_length': ref_length}
ter = evaluate.load('ter')
rouge = evaluate.load('rouge')  # ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
# meteor = evaluate.load('meteor')  # Its values range from 0 to 1
# score: BLEU score | counts: Counts | totals: Totals | precisions: Precisions | bp: Brevity penalty | sys_len: predictions length | ref_len: reference length
sacrebleu = evaluate.load('sacrebleu')


def eval(predict_text, reference_text):

    BLUE_score = bleu.compute(**{
        'predictions': [predict_text],
        'references': [reference_text]
    })
    TER_score = ter.compute(**{
        'predictions': [predict_text],
        'references': [reference_text]
    })
    ROUGE_score = rouge.compute(**{
        'predictions': [predict_text],
        'references': [reference_text]
    })
    SACREBLUE_score = sacrebleu.compute(**{
        'predictions': [predict_text],
        'references': [reference_text]
    })

    return BLUE_score, TER_score, ROUGE_score, SACREBLUE_score


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


def demo(model, tokenizer, sentence_list, num_return_sequences, num_beams, temperature, do_sample, input_string):
    dict_in_dict = {}
    model = model
    tokenizer = tokenizer
    max_sentence_output = int(num_return_sequences * len(sentence_list))
    batch = tokenizer(sentence_list, truncation=True, padding='longest',
                      max_length=100, return_tensors="pt").to(device)
    translated = model.generate(**batch, max_length=512, num_beams=num_beams,
                                num_return_sequences=num_return_sequences, temperature=temperature, do_sample=do_sample)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    for i in range(0, num_return_sequences, 1):
        merged_strings = []
        for a in range(i, max_sentence_output, num_return_sequences):
            merged_strings.append("".join(tgt_text[a]))
        paraphrased_text = " ".join(merged_strings)
        BLUE_score, TER_score, ROUGE_score, SACREBLUE_score = eval(
            predict_text=paraphrased_text, reference_text=input_string)
        dict_in_dict[i] = {
            "origin_text": f"{input_string}",
            "paraphrase_text": f"{paraphrased_text}",
            "BLUE_score": BLUE_score['bleu'],
            "TER_score": TER_score['score'],
            "ROUGE1_score": ROUGE_score['rouge1'],
            "ROUGE2_score": ROUGE_score['rouge2'],
            "ROUGEL_score": ROUGE_score['rougeL'],
            "SACREBLUE_score": SACREBLUE_score['score'],
        }

    return dict_in_dict


def demo_parrot(sentence_list, num_return_sequences, input_string):
    dict_in_dict = {}
    para_phrases = model.augment(input_phrase=sentence_list,
                                 diversity_ranker="levenshtein",
                                 do_diverse=False,
                                 max_return_phrases=num_return_sequences,
                                 max_length=32,
                                 adequacy_threshold=0.5,
                                 fluency_threshold=0.5)
    if para_phrases is not None:
        for i in para_phrases:
            BLUE_score, TER_score, ROUGE_score, SACREBLUE_score = eval(
                predict_text=i[0], reference_text=input_string)
            dict_in_dict[1] = {
                "origin_text": f"{input_string}",
                "paraphrase_text": f"{i[0]}",
                "BLUE_score": BLUE_score['bleu'],
                "TER_score": TER_score['score'],
                "ROUGE1_score": ROUGE_score['rouge1'],
                "ROUGE2_score": ROUGE_score['rouge2'],
                "ROUGEL_score": ROUGE_score['rougeL'],
                "SACREBLUE_score": SACREBLUE_score['score'],
            }
    else:
        dict_in_dict[1] = {
            "text": f"{input_string}",
            "parapharse": None,
            "BLUE_score": None,
            "TER_score": None,
            "ROUGE1_score": None,
            "ROUGEL_score": None,
            "METEOR_score": None,
            "SACREBLUE_score": None,
        }

    return dict_in_dict


def make_dataframe(dict_in_dict):
    data = []

    # Iterate over the nested dictionary
    for key, value in dict_in_dict.items():
        # Extract the relevant data from the dictionary value
        text = value['text']
        paraphrase = value['parapharse']
        BLUE_score = value['BLUE_score']
        TER_score = value['TER_score']
        ROUGE1_score = value['ROUGE1_score']
        ROUGEL_score = value['ROUGEL_score']
        METEOR_score = value['METEOR_score']
        SACREBLUE_score = value['SACREBLUE_score']
        # Create a dictionary with the extracted data
        data_row = {
            'text': text,
            'paraphrase': paraphrase,
            'BLUE_score': BLUE_score,
            'TER_score': TER_score,
            'ROUGE1_score': ROUGE1_score,
            'ROUGEL_score': ROUGEL_score,
            'METEOR_score': METEOR_score,
            'SACREBLUE_score': SACREBLUE_score
        }
        # Append the dictionary to the list
        data.append(data_row)
    # Convert the list to a DataFrame
    df = pd.DataFrame(data)
    # Print the DataFrame
    return df


_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    # st.image("logo.png", width=400)
    st.title("🔑 Pegasus - Parapraserr 🔑")
    st.header("")


with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   The *Pegasus - Parapraser* app is an easy-to-use interface built in Streamlit for the amazing [Pegasus - Parapraser](https://huggingface.co/ThanhJamieAI/ParapharseV13_10E_4B) library from HuggingFace - Transfomer!
-   It make form THEIS FPTU - HCMC - 2023 - GFA23AI15 - FA23AI12 - Phu Pham - Thanh Dang - Tri Nguyen
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## **📌 Paste sentence **")
with st.form(key="my_form"):

    ce, c1, ce, c2, c3 = st.columns([0.07, 2, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
            "Choose your model",
            ["ThanhJamieAI/ParapharseV13_10E_4B", "ThanhJamieAI/ParapharseV8_8E_4B",
                "ThanhJamieAI/ParapharseV11_4E_2B", "tuner007/pegasus_paraphrase", "Parrot"],
            help="At present, you can choose between 4 models to parapraser your sentence. More to come!",
        )

        if ModelType == "ThanhJamieAI/ParapharseV13_10E_4B":
            @st.cache(allow_output_mutation=True)
            def load_model():
                tokenizer = PegasusTokenizer.from_pretrained(ModelType)
                model = PegasusForConditionalGeneration.from_pretrained(
                    ModelType).to(device)
                return model, tokenizer
            model, tokenizer = load_model()

        if ModelType == "ThanhJamieAI/ParapharseV8_8E_4B":
            @st.cache(allow_output_mutation=True)
            def load_model():
                tokenizer = PegasusTokenizer.from_pretrained(ModelType)
                model = PegasusForConditionalGeneration.from_pretrained(
                    ModelType).to(device)
                return model, tokenizer
            model, tokenizer = load_model()

        if ModelType == "ThanhJamieAI/ParapharseV11_4E_2B":
            @st.cache(allow_output_mutation=True)
            def load_model():
                tokenizer = PegasusTokenizer.from_pretrained(ModelType)
                model = PegasusForConditionalGeneration.from_pretrained(
                    ModelType).to(device)
                return model, tokenizer
            model, tokenizer = load_model()

        if ModelType == "tuner007/pegasus_paraphrase":
            @st.cache(allow_output_mutation=True)
            def load_model():
                tokenizer = PegasusTokenizer.from_pretrained(ModelType)
                model = PegasusForConditionalGeneration.from_pretrained(
                    ModelType).to(device)
                return model, tokenizer
            model, tokenizer = load_model()

        if ModelType == "Parrot":
            @st.cache(allow_output_mutation=True)
            def load_model():
                parrot = Parrot(
                    model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)
                return parrot
            model = load_model()

        num_return_sequences = st.number_input(
            "Num_return_sequences",
            min_value=1,
            max_value=10,
            help="""The minimum value for the num_return_sequences.""",
        )

        num_beams = st.number_input(
            "Num_beams",
            value=3,
            min_value=1,
            max_value=10,
            help="""The maximum value for the Num_beams.""")

    with c2:
        doc = st.text_area(
            "Paste your text below (max 500 words)",
            height=510,
        )

        MAX_WORDS = 500
        import re
        splitter = SentenceSplitter(language='en')
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 500 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )

            doc = doc[:MAX_WORDS]
        submit_button = st.form_submit_button(label="✨ Get me the data!")
        file = st.file_uploader("Chọn file CSV", type=["csv"])
        csv = st.form_submit_button(label="Show CSV")

        # Nếu file CSV được tải lên
        if file is not None:
            # Đọc file CSV thành dataframe
            df = pd.read_csv(file)

            # Hiển thị dataframe
            st.write(df)

splitter = SentenceSplitter(language='en')
sentence_list = splitter.split(doc)
if not submit_button:
    st.stop()
if ModelType == "ThanhJamieAI/ParapharseV13_10E_4B":
    output_dict = demo(model=model, tokenizer=tokenizer, sentence_list=sentence_list,
                       num_return_sequences=num_return_sequences, num_beams=num_beams, temperature=1.5, do_sample=True, input_string=doc)
    st.write(output_dict)
if ModelType == "ThanhJamieAI/ParapharseV8_8E_4B":
    output_dict = demo(model=model, tokenizer=tokenizer, sentence_list=sentence_list,
                       num_return_sequences=num_return_sequences, num_beams=num_beams, temperature=1.5, do_sample=True, input_string=doc)
    st.write(output_dict)
if ModelType == "ThanhJamieAI/ParapharseV11_4E_2B":
    output_dict = demo(model=model, tokenizer=tokenizer, sentence_list=sentence_list,
                       num_return_sequences=num_return_sequences, num_beams=num_beams, temperature=1.5, do_sample=True, input_string=doc)
    st.write(output_dict)
if ModelType == "tuner007/pegasus_paraphrase":
    output_dict = demo(model=model, tokenizer=tokenizer, sentence_list=sentence_list,
                       num_return_sequences=num_return_sequences, num_beams=num_beams, temperature=1.5, do_sample=True, input_string=doc)
    st.write(output_dict)
if ModelType == "Parrot":
    output_dict = demo_parrot(
        sentence_list=doc, num_return_sequences=num_return_sequences, input_string=doc)
    st.write(output_dict)
