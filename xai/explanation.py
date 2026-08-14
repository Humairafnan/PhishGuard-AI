import pandas as pd


# ==========================================================
# XAI Token Cleaning
# ==========================================================

def _clean_tokens(tokens):
    """
    Remove duplicate and less meaningful tokens
    while preserving order.
    """

    STOP_WORDS = {
        "the",
        "to",
        "your",
        "a",
        "an",
        "of",
        "and",
        "is",
        "in",
        "for",
        "on",
        "with",
        "please",
        "pm",
        "am",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "0"
    }


    seen = set()
    cleaned = []


    for token in tokens:

        token_lower = str(token).lower().strip()


        if (
            token_lower not in seen
            and token_lower not in STOP_WORDS
            and len(token_lower) > 2
        ):

            seen.add(token_lower)
            cleaned.append(token)


    return cleaned



# ==========================================================
# Technical Explanation
# ==========================================================

def generate_technical_explanation(
    xai_result,
    top_n=10
):
    """
    Technical explanation for researchers/developers.

    Includes:
    - Integrated Gradients method
    - target class
    - convergence information
    - token attribution values
    """


    tokens_df = xai_result["tokens"]


    if tokens_df.empty:

        return {

            "method": "Integrated Gradients",

            "message":
                "No token attribution available.",

            "important_tokens": []

        }



    # Remove meaningless tokens for technical display also
    tokens_df = tokens_df.copy()


    tokens_df["token_clean"] = (
        tokens_df["token"]
        .astype(str)
        .str.lower()
    )


    tokens_df = tokens_df[
        ~tokens_df["token_clean"].isin(
            {
                "the",
                "to",
                "your",
                "a",
                "an",
                "of",
                "and",
                "is",
                "in",
                "for",
                "on",
                "with"
            }
        )
    ]


    top_tokens = (
        tokens_df
        .head(top_n)
    )


    important_tokens = []


    for _, row in top_tokens.iterrows():

        effect = (

            "supports phishing"

            if row["importance"] > 0

            else

            "opposes phishing"

        )


        important_tokens.append(

            {

                "token": row["token"],


                "importance": round(
                    float(row["importance"]),
                    6
                ),


                "relative_importance": round(
                    float(row["relative_importance"]),
                    6
                ),


                "effect": effect

            }

        )



    return {

        "method":
            "Integrated Gradients",


        "target":
            "Phishing-class logit",


        "convergence_delta":
            round(
                float(
                    xai_result["convergence_delta"]
                ),
                6
            ),


        "steps_used":
            xai_result["steps_used"],


        "converged":
            xai_result["converged"],


        "important_tokens":
            important_tokens,


        "interpretation":
            (
                "Positive attribution values increase "
                "the phishing-class score, while negative "
                "values reduce the phishing-class score."
            )

    }





# ==========================================================
# User Explanation
# ==========================================================

def generate_user_explanation(
    xai_result,
    prediction,
    top_n=5
):
    """
    User-friendly explanation for general users.
    """


    tokens_df = xai_result["tokens"]


    if tokens_df.empty:

        return {

            "summary":
                "The AI could not identify specific "
                "textual factors influencing this decision.",


            "positive_factors": [],


            "negative_factors": []

        }




    # -----------------------------
    # Positive phishing contributors
    # -----------------------------

    positive_tokens = (

        tokens_df[
            tokens_df["importance"] > 0
        ]

        .sort_values(
            "abs_importance",
            ascending=False
        )

        ["token"]

        .tolist()

    )



    # -----------------------------
    # Negative phishing contributors
    # -----------------------------

    negative_tokens = (

        tokens_df[
            tokens_df["importance"] < 0
        ]

        .sort_values(
            "abs_importance",
            ascending=False
        )

        ["token"]

        .tolist()

    )



    # Clean tokens

    positive_tokens = (

        _clean_tokens(
            positive_tokens
        )

        [:top_n]

    )



    negative_tokens = (

        _clean_tokens(
            negative_tokens
        )

        [:top_n]

    )




    # -----------------------------
    # Prediction explanation
    # -----------------------------

    if str(prediction).lower() in [
        "phishing",
        "1"
    ]:


        summary = (

            "⚠️ This email appears suspicious. "
            "The AI detected language patterns "
            "commonly associated with phishing attempts."

        )


        if positive_tokens:

            summary += (

                " Important indicators that increased "
                "the phishing likelihood include: "
                +

                ", ".join(
                    positive_tokens
                )

                +

                "."

            )



    else:


        summary = (

            "✅ This email appears legitimate. "
            "The AI detected patterns that are "
            "more consistent with normal email communication."

        )


        if negative_tokens:

            summary += (

                " Factors that reduced the phishing "
                "likelihood include: "

                +

                ", ".join(
                    negative_tokens
                )

                +

                "."

            )





    return {

        "summary":
            summary,


        "positive_factors":
            positive_tokens,


        "negative_factors":
            negative_tokens

    }