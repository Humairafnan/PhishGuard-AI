from flask import Flask, render_template, request, session

import os

from dotenv import load_dotenv

from utils.confidence import get_risk_level

from model.inference import predict_email

from xai.integrated_gradients import explain_email

from xai.explanation import (
    generate_technical_explanation,
    generate_user_explanation
)

from utils.xai_visualization import create_xai_plot


load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "temporary-secret-key"
)





# ==========================
# HOME PAGE
# ==========================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )






# ==========================
# ANALYZE EMAIL
# ==========================


@app.route(
    "/analyze",
    methods=["GET", "POST"]
)
def analyze():

    result = None

    risk = None



    if request.method == "POST":


        email_text = request.form.get(
            "email_text"
        )



        if not email_text or not email_text.strip():

            return render_template(
                "analyze.html",
                error="Please enter email content."
            )



        try:


            # Store email for XAI
            session["email_text"] = email_text



            # ----------------------------
            # Prediction
            # ----------------------------

            result = predict_email(
                email_text
            )



            # Store URL analysis
            session["url_analysis"] = (
                result["url_analysis"]
            )



            # Store prediction
            session["prediction_result"] = result





            # ----------------------------
            # Risk Classification
            # ----------------------------

            risk = get_risk_level(

                result["legitimate_probability"],

                result["phishing_probability"],

                result["url_analysis"]

            )



            session["risk_result"] = risk





        except Exception as e:


            print(
                "Analysis failed:",
                e
            )


            return render_template(

                "analyze.html",

                error=str(e)

            )




    return render_template(

        "analyze.html",

        result=result,

        risk=risk

    )









# ==========================
# XAI EXPLANATION
# ==========================


@app.route("/explain")
def explain():


    user_explanation = None

    technical_explanation = None

    xai_plot = None




    email_text = session.get(
        "email_text"
    )



    url_analysis = session.get(
        "url_analysis"
    )



    prediction_result = session.get(
        "prediction_result"
    )



    risk = session.get(
        "risk_result"
    )





    if not email_text or not prediction_result:


        return render_template(

            "analyze.html",

            error="No email available for explanation."

        )






    try:



        # ----------------------------
        # Integrated Gradients
        # ----------------------------

        xai_result = explain_email(
            email_text
        )





        # ----------------------------
        # User Explanation
        # ----------------------------

        user_explanation = generate_user_explanation(

            xai_result,

            prediction_result["prediction"]

        )






        # ----------------------------
        # Technical Explanation
        # ----------------------------

        technical_explanation = generate_technical_explanation(

            xai_result

        )






        # ----------------------------
        # XAI Visualization
        # ----------------------------

        try:


            xai_plot = create_xai_plot(

                technical_explanation

            )


        except Exception as plot_error:


            print(

                "XAI plot generation failed:",

                plot_error

            )


            xai_plot = None






    except Exception as e:


        print(

            "Explanation failed:",

            e

        )



        return render_template(

            "analyze.html",

            error=str(e)

        )







    return render_template(

        "analyze.html",

        user_explanation=user_explanation,

        technical_explanation=technical_explanation,

        xai_plot=xai_plot,

        url_analysis=url_analysis,

        risk=risk

    )








# ==========================
# HOW IT WORKS PAGE
# ==========================


@app.route("/how-it-works")
def how_it_works():

    return render_template(
        "how_it_works.html"
    )



@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )