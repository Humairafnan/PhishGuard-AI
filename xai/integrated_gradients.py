import re
import gc
import torch
import pandas as pd

from torch import nn
from captum.attr import IntegratedGradients

from model.loader import (
    model,
    tokenizer,
    DEVICE
)


MAX_LENGTH = 128

PHISHING_CLASS = 1

# Deployment XAI setting
N_STEPS = 16

CONVERGENCE_TOLERANCE = 0.1

INTERNAL_BATCH_SIZE = 8



class PhishingLogitWrapper(nn.Module):

    def __init__(self, model):

        super().__init__()

        self.model = model



    def forward(
        self,
        inputs_embeds,
        attention_mask
    ):


        outputs = self.model.electra(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )


        hidden_states = outputs.last_hidden_state


        cls_vector = hidden_states[:,0,:]


        x = hidden_states.permute(
            0,
            2,
            1
        )


        conv3 = self.model.relu(
            self.model.conv1(x)
        )


        conv5 = self.model.relu(
            self.model.conv2(x)
        )


        pool3 = torch.max(
            conv3,
            dim=2
        )[0]


        pool5 = torch.max(
            conv5,
            dim=2
        )[0]


        cnn_features = torch.cat(
            [
                pool3,
                pool5
            ],
            dim=1
        )


        combined = torch.cat(
            [
                cls_vector,
                cnn_features
            ],
            dim=1
        )


        combined = self.model.dropout(
            combined
        )


        logits = self.model.classifier(
            combined
        )


        return logits[:, PHISHING_CLASS]



xai_wrapper = PhishingLogitWrapper(
    model
).to(DEVICE)



ig = IntegratedGradients(
    xai_wrapper
)



def make_baseline_ids(input_ids):

    pad_id = tokenizer.pad_token_id


    if pad_id is None:

        raise ValueError(
            "Tokenizer has no pad token."
        )


    baseline_ids = torch.full_like(
        input_ids,
        fill_value=pad_id
    )


    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id


    if cls_id is not None:

        baseline_ids[input_ids == cls_id] = cls_id


    if sep_id is not None:

        baseline_ids[input_ids == sep_id] = sep_id


    return baseline_ids




SPECIAL_TOKENS = {

    tokenizer.cls_token,
    tokenizer.sep_token,
    tokenizer.pad_token,
    tokenizer.unk_token

}




def merge_wordpieces_in_order(
    tokens,
    scores,
    attention_mask
):

    rows = []


    current_token = None
    current_score = 0.0
    current_start = None
    current_end = None



    def flush():

        nonlocal current_token
        nonlocal current_score
        nonlocal current_start
        nonlocal current_end


        if current_token is not None:

            token = current_token.strip()


            if (
                token
                and token not in SPECIAL_TOKENS
                and re.search(r"[A-Za-z0-9]", token)
            ):

                rows.append(
                    {
                        "token": token,
                        "importance": float(current_score),
                        "start_position": int(current_start),
                        "end_position": int(current_end)
                    }
                )


        current_token = None
        current_score = 0.0
        current_start = None
        current_end = None



    for pos,(token,score,mask_value) in enumerate(
        zip(tokens,scores,attention_mask)
    ):


        if int(mask_value)==0:

            flush()
            continue



        if token in SPECIAL_TOKENS:

            flush()
            continue



        if token.startswith("##") and current_token is not None:

            current_token += token[2:]

            current_score += float(score)

            current_end = pos


        else:

            flush()

            current_token = (
                token[2:]
                if token.startswith("##")
                else token
            )

            current_score=float(score)

            current_start=pos

            current_end=pos



    flush()


    result=pd.DataFrame(rows)


    if len(result)>0:

        result["abs_importance"] = (
            result["importance"]
            .abs()
        )


    return result





def explain_email(text):


    if (
        not isinstance(text,str)
        or not text.strip()
    ):

        raise ValueError(
            "Email text must be non-empty."
        )



    model.eval()



    encoding = tokenizer(

        text,

        truncation=True,

        padding="max_length",

        max_length=MAX_LENGTH,

        return_tensors="pt"

    )



    input_ids = encoding["input_ids"].to(
        DEVICE
    )


    attention_mask = encoding["attention_mask"].to(
        DEVICE
    )



    baseline_ids = make_baseline_ids(
        input_ids
    )



    embedding_layer = (
        model.electra
        .embeddings
        .word_embeddings
    )


    input_embeddings = embedding_layer(
        input_ids
    )


    baseline_embeddings = embedding_layer(
        baseline_ids
    )



    print(
        f"Running IG with {N_STEPS} steps..."
    )



    attributions, delta = ig.attribute(

        inputs=input_embeddings,

        baselines=baseline_embeddings,

        additional_forward_args=(
            attention_mask,
        ),

        n_steps=N_STEPS,

        internal_batch_size=INTERNAL_BATCH_SIZE,

        return_convergence_delta=True

    )



    print(
        "IG attribution completed"
    )



    convergence_delta=float(
        delta.detach()
        .cpu()
        .reshape(-1)[0]
        .item()
    )



    scores = (
        attributions
        .sum(dim=-1)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )



    token_ids = (
        input_ids
        .squeeze(0)
        .detach()
        .cpu()
        .tolist()
    )


    tokens = tokenizer.convert_ids_to_tokens(
        token_ids
    )


    mask_values = (
        attention_mask
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )



    token_df = merge_wordpieces_in_order(
        tokens,
        scores,
        mask_values
    )



    if len(token_df)>0:


        total_abs = (
            token_df["abs_importance"]
            .sum()
        )


        if total_abs>0:

            token_df["relative_importance"] = (
                token_df["abs_importance"]
                /
                total_abs
            )

        else:

            token_df["relative_importance"]=0.0



        token_df = (
            token_df
            .sort_values(
                "abs_importance",
                ascending=False
            )
            .reset_index(drop=True)
        )



    result = {

        "tokens":token_df,

        "convergence_delta":convergence_delta,

        "steps_used":N_STEPS,

        "converged":
        abs(convergence_delta)
        <= CONVERGENCE_TOLERANCE

    }


    # Memory cleanup
    del encoding
    del input_ids
    del attention_mask
    del baseline_ids
    del input_embeddings
    del baseline_embeddings
    del attributions


    gc.collect()


    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()



    return result