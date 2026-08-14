import matplotlib

# Use non-GUI backend for Flask/server environments
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import os



def create_xai_plot(
    technical_explanation,
    filename="xai_plot.png"
):


    tokens = []

    values = []



    for item in technical_explanation["important_tokens"]:


        tokens.append(
            item["token"]
        )


        values.append(
            item["importance"]
        )



    # Create figure

    plt.figure(
        figsize=(8, 5)
    )



    plt.barh(

        tokens[::-1],

        values[::-1]

    )



    plt.axvline(

        0,

        linewidth=1

    )



    plt.xlabel(
        "Attribution Score"
    )


    plt.ylabel(
        "Token"
    )


    plt.title(
        "Integrated Gradients Token Attribution"
    )



    plt.tight_layout()



    output_path = os.path.join(

        "static",

        "images",

        filename

    )



    # Ensure directory exists

    os.makedirs(

        os.path.dirname(output_path),

        exist_ok=True

    )



    plt.savefig(

        output_path,

        bbox_inches="tight",

        dpi=300

    )



    # Important for Flask deployment

    plt.close(
        "all"
    )



    return filename