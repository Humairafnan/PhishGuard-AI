import torch

from model.loader import model, tokenizer, DEVICE

from utils.url_analysis import analyze_urls



MAX_LENGTH = 256



LABEL_NAMES = {
    0: "Legitimate",
    1: "Phishing"
}





def predict_email(email_text):
    """
    Run phishing prediction on a single email.

    Combines:
    - ELECTRA-CNN text prediction
    - URL security analysis
    """



    if not email_text or not email_text.strip():

        raise ValueError(
            "Email text is empty."
        )



    model.eval()



    # ---------------------------------
    # URL Analysis
    # ---------------------------------

    url_analysis = analyze_urls(
        email_text
    )



    # ---------------------------------
    # Tokenize input
    # ---------------------------------

    inputs = tokenizer(

        email_text,

        return_tensors="pt",

        truncation=True,

        padding="max_length",

        max_length=MAX_LENGTH

    )



    input_ids = inputs["input_ids"].to(
        DEVICE
    )


    attention_mask = inputs["attention_mask"].to(
        DEVICE
    )




    # ---------------------------------
    # Model Prediction
    # ---------------------------------

    with torch.no_grad():


        logits = model(

            input_ids=input_ids,

            attention_mask=attention_mask

        )



        probabilities = torch.softmax(

            logits,

            dim=1

        )[0]




    predicted_class = int(

        torch.argmax(
            probabilities
        ).item()

    )




    legitimate_probability = float(

        probabilities[0].item()

    )



    phishing_probability = float(

        probabilities[1].item()

    )



    confidence = float(

        probabilities[predicted_class].item()

    )




    # ---------------------------------
    # Final Result
    # ---------------------------------

    result = {


        "prediction":

            LABEL_NAMES[predicted_class],



        "predicted_class":

            predicted_class,



        "legitimate_probability":

            legitimate_probability,



        "phishing_probability":

            phishing_probability,



        "confidence":

            confidence,



        "url_analysis":

            url_analysis

    }




    # ---------------------------------
    # Release temporary tensors
    # ---------------------------------

    del inputs

    del input_ids

    del attention_mask

    del logits

    del probabilities



    return result