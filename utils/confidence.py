def get_risk_level(
    legitimate_probability,
    phishing_probability,
    url_analysis=None
):
    """
    Determine final email risk level.

    Uses:
    - ELECTRA-CNN prediction probabilities
    - URL security analysis

    URL analysis adjusts risk assessment
    but does not change model prediction.
    """



    difference = abs(
        legitimate_probability - phishing_probability
    )



    # ---------------------------------
    # URL information
    # ---------------------------------

    url_score = 0
    urls_found = 0


    if url_analysis:

        url_score = url_analysis.get(
            "total_risk_score",
            0
        )

        urls_found = url_analysis.get(
            "urls_found",
            0
        )




    # ---------------------------------
    # Strong suspicious URLs
    # ---------------------------------

    if url_score >= 4:

        if phishing_probability >= 0.60:

            return {
                "level": "High Risk Phishing",
                "category": "high_risk_phishing",
                "message":
                "The email contains suspicious language patterns and potentially unsafe URLs."
            }



    # ---------------------------------
    # Trusted URLs reduce uncertainty
    # ---------------------------------

    if url_score < 0 and phishing_probability >= 0.60:

        return {
            "level": "Uncertain",
            "category": "uncertain",
            "message":
            "The email contains suspicious language patterns, but detected URLs appear trustworthy. Manual verification is recommended."
        }





    # ---------------------------------
    # Very close probabilities
    # ---------------------------------

    if difference < 0.15:

        return {
            "level": "Uncertain",
            "category": "uncertain",
            "message":
            "The email contains both legitimate and suspicious patterns. Manual review is recommended."
        }





    # ---------------------------------
    # Legitimate cases
    # ---------------------------------

    if legitimate_probability >= 0.85:


        return {
            "level": "Legitimate",
            "category": "legitimate",
            "message":
            "The email appears safe with strong legitimate indicators."
        }




    elif legitimate_probability >= 0.60:


        return {
            "level": "Likely Legitimate",
            "category": "likely_legitimate",
            "message":
            "The email appears mostly legitimate but contains some ambiguous patterns."
        }






    # ---------------------------------
    # Phishing cases
    # ---------------------------------

    if phishing_probability >= 0.85:


        return {
            "level": "High Risk Phishing",
            "category": "high_risk_phishing",
            "message":
            "The email contains strong indicators commonly associated with phishing."
        }




    elif phishing_probability >= 0.60:


        return {
            "level": "Likely Phishing",
            "category": "likely_phishing",
            "message":
            "The email contains suspicious patterns that require caution."
        }





    return {
        "level": "Uncertain",
        "category": "uncertain",
        "message":
        "The model could not confidently determine the email risk level."
    }