#@title Helper Functions

import glob
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from IPython.display import display
import PIL
import fitz
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from vertexai.generative_models import (
    GenerationConfig,
    HarmBlockThreshold,
    HarmCategory,
    Image,
)
from vertexai.vision_models import Image as vision_model_Image
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel

# function to set embeddings as global variable

# multimodal_model_2_0_flash = GenerativeModel(
#     "gemini-2.0-flash-001"
# ) # Gemini latest Gemini 2.0 Flash Model


# multimodal_model_2_0_flash = GenerativeModel(
#     "gemini-2.0-flash-001"
# ) # Gemini latest Gemini 2.0 Flash Model

# multimodal_model_15 = GenerativeModel(
#     "gemini-1.5-pro-002"
# )  # works with text, code, images, video(with or without audio) and audio(mp3) with 1M input context - complex reasoning


# Load text embedding model from pre-trained source
text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# Load multimodal embedding model from pre-trained source
multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(
    "multimodalembedding@001"
)  # works with image, image with caption(~32 words), video, video with caption(~32 words)

# multimodal_model_15_flash = GenerativeModel(
#     "gemini-1.5-flash-002"
# )  # works with text, code, images, video(with or without audio) and audio(mp3) with 1M input context - faster inference



def set_global_variable(variable_name: str, value: any) -> None:
    """
    Sets the value of a global variable.

    Args:
        variable_name: The name of the global variable (as a string).
        value: The value to assign to the global variable. This can be of any type.
    """
    global_vars = globals()  # Get a dictionary of global variables
    global_vars[variable_name] = value

# Functions for getting text and image embeddings
def get_text_embedding_from_text_embedding_model(
    text: str,
    return_array: Optional[bool] = False,
) -> list:
    """
    Generates a numerical text embedding from a provided text input using a text embedding model.

    Args:
        text: The input text string to be embedded.
        return_array: If True, returns the embedding as a NumPy array.
                      If False, returns the embedding as a list. (Default: False)

    Returns:
        list or numpy.ndarray: A 768-dimensional vector representation of the input text.
                               The format (list or NumPy array) depends on the
                               value of the 'return_array' parameter.
    """
    embeddings = text_embedding_model.get_embeddings([text])
    text_embedding = [embedding.values for embedding in embeddings][0]

    if return_array:
        text_embedding = np.fromiter(text_embedding, dtype=float)

    # returns 768 dimensional array
    return text_embedding


def get_image_embedding_from_multimodal_embedding_model(
    image_uri: str,
    embedding_size: int = 512,
    text: Optional[str] = None,
    return_array: Optional[bool] = False,
) -> list:
    """Extracts an image embedding from a multimodal embedding model.
    The function can optionally utilize contextual text to refine the embedding.

    Args:
        image_uri (str): The URI (Uniform Resource Identifier) of the image to process.
        text (Optional[str]): Optional contextual text to guide the embedding generation. Defaults to "".
        embedding_size (int): The desired dimensionality of the output embedding. Defaults to 512.
        return_array (Optional[bool]): If True, returns the embedding as a NumPy array.
        Otherwise, returns a list. Defaults to False.

    Returns:
        list: A list containing the image embedding values. If `return_array` is True, returns a NumPy array instead.
    """
    # image = Image.load_from_file(image_uri)
    image = vision_model_Image.load_from_file(image_uri)
    embeddings = multimodal_embedding_model.get_embeddings(
        image=image, contextual_text=text, dimension=embedding_size
    )  # 128, 256, 512, 1408
    image_embedding = embeddings.image_embedding

    if return_array:
        image_embedding = np.fromiter(image_embedding, dtype=float)

    return image_embedding


def get_gemini_response(
    generative_multimodal_model,
    model_input: List[str],
    stream: bool = True,
    generation_config: Optional[GenerationConfig] = GenerationConfig(
        temperature=0.2, max_output_tokens=2048
    ),
    safety_settings: Optional[dict] = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
    print_exception: bool = False,
) -> str:
    """
    This function generates text in response to a list of model inputs.

    Args:
        model_input: A list of strings representing the inputs to the model.
        stream: Whether to generate the response in a streaming fashion (returning chunks of text at a time) or all at once. Defaults to False.

    Returns:
        The generated text as a string.
    """
    response = generative_multimodal_model.generate_content(
        model_input,
        generation_config=generation_config,
        stream=stream,
        safety_settings=safety_settings,
    )
    response_list = []

    for chunk in response:
        try:
            response_list.append(chunk.text)
        except Exception as e:
            if print_exception:
              print(
                  "Exception occurred while calling gemini. Something is blocked. Lower the safety thresholds [safety_settings: BLOCK_NONE ] if not already done. -----",
                  e,
              )
            else:
              print("Exception occurred while calling gemini. Something is blocked. Lower the safety thresholds [safety_settings: BLOCK_NONE ] if not already done. -----")
            response_list.append("**Something blocked.**")
            continue
    response = "".join(response_list)

    return response

def get_user_query_text_embeddings(user_query: str) -> np.ndarray:
    """
    Extracts text embeddings for the user query using a text embedding model.

    Args:
        user_query: The user query text.
        embedding_size: The desired embedding size.

    Returns:
        A NumPy array representing the user query text embedding.
    """

    return get_text_embedding_from_text_embedding_model(user_query)


def get_user_query_image_embeddings(
    image_query_path: str, embedding_size: int
) -> np.ndarray:
    """
    Extracts image embeddings for the user query image using a multimodal embedding model.

    Args:
        image_query_path: The path to the user query image.
        embedding_size: The desired embedding size.

    Returns:
        A NumPy array representing the user query image embedding.
    """

    return get_image_embedding_from_multimodal_embedding_model(
        image_uri=image_query_path, embedding_size=embedding_size
    )


def get_cosine_score(
    dataframe: pd.DataFrame, column_name: str, input_text_embd: np.ndarray
) -> float:
    """
    Calculates the cosine similarity between the user query embedding and the dataframe embedding for a specific column.

    Args:
        dataframe: The pandas DataFrame containing the data to compare against.
        column_name: The name of the column containing the embeddings to compare with.
        input_text_embd: The NumPy array representing the user query embedding.

    Returns:
        The cosine similarity score (rounded to two decimal places) between the user query embedding and the dataframe embedding.
    """

    text_cosine_score = round(np.dot(dataframe[column_name], input_text_embd), 2)
    return text_cosine_score


def print_text_to_image_citation(
    final_images: Dict[int, Dict[str, Any]], print_top: bool = True
) -> None:
    """
    Prints a formatted citation for each matched image in a dictionary.

    Args:
        final_images: A dictionary containing information about matched images,
                    with keys as image number and values as dictionaries containing
                    image path, page number, page text, cosine similarity score, and image description.
        print_top: A boolean flag indicating whether to only print the first citation (True) or all citations (False).

    Returns:
        None (prints formatted citations to the console).
    """

    color = Color()

    # Iterate through the matched image citations
    for imageno, image_dict in final_images.items():
        # Print the citation header
        print(
            color.RED + f"Citation {imageno + 1}:",
            "Matched image path, page number and page text: \n" + color.END,
        )

        # Print the cosine similarity score
        print(color.BLUE + "score: " + color.END, image_dict["cosine_score"])

        # Print the file_name
        print(color.BLUE + "file_name: " + color.END, image_dict["file_name"])

        # Print the image path
        print(color.BLUE + "path: " + color.END, image_dict["img_path"])

        # Print the page number
        print(color.BLUE + "page number: " + color.END, image_dict["page_num"])

        # Print the page text
        print(
            color.BLUE + "page text: " + color.END, "\n".join(image_dict["page_text"])
        )

        # Print the image description
        print(
            color.BLUE + "image description: " + color.END,
            image_dict["image_description"],
        )

        # Only print the first citation if print_top is True
        if print_top and imageno == 0:
            break


def print_text_to_text_citation(
    final_text: Dict[int, Dict[str, Any]],
    print_top: bool = True,
    chunk_text: bool = True,
) -> None:
    """
    Prints a formatted citation for each matched text in a dictionary.

    Args:
        final_text: A dictionary containing information about matched text passages,
                    with keys as text number and values as dictionaries containing
                    page number, cosine similarity score, chunk number (optional),
                    chunk text (optional), and page text (optional).
        print_top: A boolean flag indicating whether to only print the first citation (True) or all citations (False).
        chunk_text: A boolean flag indicating whether to print individual text chunks (True) or the entire page text (False).

    Returns:
        None (prints formatted citations to the console).
    """

    color = Color()

    # Iterate through the matched text citations
    for textno, text_dict in final_text.items():
        # Print the citation header
        print(color.RED + f"Citation {textno + 1}:", "Matched text: \n" + color.END)

        # Print the cosine similarity score
        print(color.BLUE + "score: " + color.END, text_dict["cosine_score"])

        # Print the file_name
        print(color.BLUE + "file_name: " + color.END, text_dict["file_name"])

        # Print the page number
        print(color.BLUE + "page_number: " + color.END, text_dict["page_num"])

        # Print the matched text based on the chunk_text argument
        if chunk_text:
            # Print chunk number and chunk text
            print(color.BLUE + "chunk_number: " + color.END, text_dict["chunk_number"])
            print(color.BLUE + "chunk_text: " + color.END, text_dict["chunk_text"])
        else:
            # Print page text
            print(color.BLUE + "page text: " + color.END, text_dict["page_text"])

        # Only print the first citation if print_top is True
        if print_top and textno == 0:
            break


def get_similar_image_from_query(
    text_metadata_df: pd.DataFrame,
    image_metadata_df: pd.DataFrame,
    query: str = "",
    image_query_path: str = "",
    column_name: str = "",
    image_emb: bool = True,
    top_n: int = 3,
    embedding_size: int = 128,
) -> Dict[int, Dict[str, Any]]:
    """
    Finds the top N most similar images from a metadata DataFrame based on a text query or an image query.

    Args:
        text_metadata_df: A Pandas DataFrame containing text metadata associated with the images.
        image_metadata_df: A Pandas DataFrame containing image metadata (paths, descriptions, etc.).
        query: The text query used for finding similar images (if image_emb is False).
        image_query_path: The path to the image used for finding similar images (if image_emb is True).
        column_name: The column name in the image_metadata_df containing the image embeddings or captions.
        image_emb: Whether to use image embeddings (True) or text captions (False) for comparisons.
        top_n: The number of most similar images to return.
        embedding_size: The dimensionality of the image embeddings (only used if image_emb is True).

    Returns:
        A dictionary containing information about the top N most similar images, including cosine scores, image objects, paths, page numbers, text excerpts, and descriptions.
    """
    # Check if image embedding is used
    if image_emb:
        # Calculate cosine similarity between query image and metadata images
        user_query_image_embedding = get_user_query_image_embeddings(
            image_query_path, embedding_size
        )
        cosine_scores = image_metadata_df.apply(
            lambda x: get_cosine_score(x, column_name, user_query_image_embedding),
            axis=1,
        )
    else:
        # Calculate cosine similarity between query text and metadata image captions
        user_query_text_embedding = get_user_query_text_embeddings(query)
        cosine_scores = image_metadata_df.apply(
            lambda x: get_cosine_score(x, column_name, user_query_text_embedding),
            axis=1,
        )

    # Remove same image comparison score when user image is matched exactly with metadata image
    cosine_scores = cosine_scores[cosine_scores < 1.0]

    # Get top N cosine scores and their indices
    top_n_cosine_scores = cosine_scores.nlargest(top_n).index.tolist()
    top_n_cosine_values = cosine_scores.nlargest(top_n).values.tolist()

    # Create a dictionary to store matched images and their information
    final_images: Dict[int, Dict[str, Any]] = {}

    for matched_imageno, indexvalue in enumerate(top_n_cosine_scores):
        # Create a sub-dictionary for each matched image
        final_images[matched_imageno] = {}

        # Store cosine score
        final_images[matched_imageno]["cosine_score"] = top_n_cosine_values[
            matched_imageno
        ]

        # Load image from file
        final_images[matched_imageno]["image_object"] = Image.load_from_file(
            image_metadata_df.iloc[indexvalue]["img_path"]
        )

        # Add file name
        final_images[matched_imageno]["file_name"] = image_metadata_df.iloc[indexvalue][
            "file_name"
        ]

        # Store image path
        final_images[matched_imageno]["img_path"] = image_metadata_df.iloc[indexvalue][
            "img_path"
        ]

        # Store page number
        final_images[matched_imageno]["page_num"] = image_metadata_df.iloc[indexvalue][
            "page_num"
        ]

        final_images[matched_imageno]["page_text"] = np.unique(
            text_metadata_df[
                (
                    text_metadata_df["page_num"].isin(
                        [final_images[matched_imageno]["page_num"]]
                    )
                )
                & (
                    text_metadata_df["file_name"].isin(
                        [final_images[matched_imageno]["file_name"]]
                    )
                )
            ]["text"].values
        )

        # Store image description
        final_images[matched_imageno]["image_description"] = image_metadata_df.iloc[
            indexvalue
        ]["img_desc"]

    return final_images


def get_similar_text_from_query(
    query: str,
    text_metadata_df: pd.DataFrame,
    column_name: str = "",
    top_n: int = 3,
    chunk_text: bool = True,
    print_citation: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Finds the top N most similar text passages from a metadata DataFrame based on a text query.

    Args:
        query: The text query used for finding similar passages.
        text_metadata_df: A Pandas DataFrame containing the text metadata to search.
        column_name: The column name in the text_metadata_df containing the text embeddings or text itself.
        top_n: The number of most similar text passages to return.
        embedding_size: The dimensionality of the text embeddings (only used if text embeddings are stored in the column specified by `column_name`).
        chunk_text: Whether to return individual text chunks (True) or the entire page text (False).
        print_citation: Whether to immediately print formatted citations for the matched text passages (True) or just return the dictionary (False).

    Returns:
        A dictionary containing information about the top N most similar text passages, including cosine scores, page numbers, chunk numbers (optional), and chunk text or page text (depending on `chunk_text`).

    Raises:
        KeyError: If the specified `column_name` is not present in the `text_metadata_df`.
    """

    if column_name not in text_metadata_df.columns:
        raise KeyError(f"Column '{column_name}' not found in the 'text_metadata_df'")

    query_vector = get_user_query_text_embeddings(query)

    # Calculate cosine similarity between query text and metadata text
    cosine_scores = text_metadata_df.apply(
        lambda row: get_cosine_score(
            row,
            column_name,
            query_vector,
        ),
        axis=1,
    )

    # Get top N cosine scores and their indices
    top_n_indices = cosine_scores.nlargest(top_n).index.tolist()
    top_n_scores = cosine_scores.nlargest(top_n).values.tolist()

    # Create a dictionary to store matched text and their information
    final_text: Dict[int, Dict[str, Any]] = {}

    for matched_textno, index in enumerate(top_n_indices):
        # Create a sub-dictionary for each matched text
        final_text[matched_textno] = {}

        # Store page number
        final_text[matched_textno]["file_name"] = text_metadata_df.iloc[index][
            "file_name"
        ]

        # Store page number
        final_text[matched_textno]["page_num"] = text_metadata_df.iloc[index][
            "page_num"
        ]

        # Store cosine score
        final_text[matched_textno]["cosine_score"] = top_n_scores[matched_textno]

        if chunk_text:
            # Store chunk number
            final_text[matched_textno]["chunk_number"] = text_metadata_df.iloc[index][
                "chunk_number"
            ]

            # Store chunk text
            final_text[matched_textno]["chunk_text"] = text_metadata_df["chunk_text"][
                index
            ]
        else:
            # Store page text
            final_text[matched_textno]["text"] = text_metadata_df["text"][index]

    # Optionally print citations immediately
    if print_citation:
        print_text_to_text_citation(final_text, chunk_text=chunk_text)

    return final_text


def display_images(
    images: Iterable[Union[str, PIL.Image.Image]], resize_ratio: float = 0.5
) -> None:
    """
    Displays a series of images provided as paths or PIL Image objects.

    Args:
        images: An iterable of image paths or PIL Image objects.
        resize_ratio: The factor by which to resize each image (default 0.5).

    Returns:
        None (displays images using IPython or Jupyter notebook).
    """

    # Convert paths to PIL images if necessary
    pil_images = []
    for image in images:
        if isinstance(image, str):
            pil_images.append(PIL.Image.open(image))
        else:
            pil_images.append(image)

    # Resize and display each image
    for img in pil_images:
        original_width, original_height = img.size
        new_width = int(original_width * resize_ratio)
        new_height = int(original_height * resize_ratio)
        resized_img = img.resize((new_width, new_height))
        display(resized_img)
        print("\n")

def get_answer_from_qa_system(
    query: str,
    text_metadata_df,
    image_metadata_df,
    top_n_text: int = 10,
    top_n_image: int = 5,
    instruction: Optional[str] = None,
    model=None,
    generation_config: Optional[GenerationConfig] = GenerationConfig(
        temperature=1, max_output_tokens=8192
    ),
    safety_settings: Optional[dict] = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
) -> Union[str, None]:
    """Fetches answers from a combined text and image-based QA system.

    Args:
        query (str): The user's question.
        text_metadata_df: DataFrame containing text embeddings, file names, and page numbers.
        image_metadata_df: DataFrame containing image embeddings, paths, and descriptions.
        top_n_text (int, optional): Number of top text chunks to consider. Defaults to 10.
        top_n_image (int, optional): Number of top images to consider. Defaults to 5.
        instruction (str, optional): Customized instruction for the model. Defaults to a generic one.
        model: Model to use for QA.
        safety_settings: Safety settings for the model.
        generation_config: Generation configuration for the model.

    Returns:
        Union[str, None]: The generated answer or None if an error occurs.
    """
    # Build Gemini content
    if instruction is None:  # Use default instruction if not provided
        instruction = """Task: Answer the following questions in detail, providing clear reasoning and evidence from the images and text in bullet points.
                      Instructions:

                      1. **Analyze:** Carefully examine the provided images and text context.
                      2. **Synthesize:** Integrate information from both the visual and textual elements.
                      3. **Reason:**  Deduce logical connections and inferences to address the question.
                      4. **Respond:** Provide a concise, accurate answer in the following format:

                        * **Question:** [Question]
                        * **Answer:** [Direct response to the question]
                        * **Explanation:** [Bullet-point reasoning steps if applicable]
                        * **Source** [name of the file, page, image from where the information is citied]

                      5. **Ambiguity:** If the context is insufficient to answer, make a reasonable guess but give the full reasoning.
                      6. Do not make up any information.
                      7. so "earning call documents" here only refer to the .pdf files.
                      8. when you are asked to count something, do not make up any information. and try to count seperately, like how many .pdf files, or how many .png files, etc. 
                      """

    # Retrieve relevant chunks of text based on the query
    matching_results_chunks_data = get_similar_text_from_query(
        query,
        text_metadata_df,
        column_name="text_embedding_chunk",
        top_n=top_n_text,
        chunk_text=True,
    )
    # Get all relevant images based on user query
    matching_results_image_fromdescription_data = get_similar_image_from_query(
        text_metadata_df,
        image_metadata_df,
        query=query,
        column_name="text_embedding_from_image_description",
        image_emb=False,
        top_n=top_n_image,
        embedding_size=1408,
    )

    # combine all the selected relevant text chunks
    context_text = ["Text Context: "]
    for key, value in matching_results_chunks_data.items():
        context_text.extend(
            [
                "Text Source: ",
                f"""file_name: "{value["file_name"]}" Page: "{value["page_num"]}""",
                "Text",
                value["chunk_text"],
            ]
        )

    # combine all the selected relevant images
    gemini_content = [
        instruction,
        "Questions: ",
        query,
        "Image Context: ",
    ]
    for key, value in matching_results_image_fromdescription_data.items():
        gemini_content.extend(
            [
                "Image Path: ",
                value["img_path"],
                "Image Description: ",
                value["image_description"],
                "Image:",
                value["image_object"],
            ]
        )
    gemini_content.extend(context_text)

    # Get Gemini response with streaming (if supported)
    response = get_gemini_response(
        model,
        model_input=gemini_content,
        stream=True,
        safety_settings=safety_settings,
        generation_config=generation_config,
    )

    return (
        response,
        matching_results_chunks_data,
        matching_results_image_fromdescription_data,
    )

def upload_to_gcs(local_path, bucket="ifad-lanzi-mrag-food", folder="parquet"):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(f"{folder}/{Path(local_path).name}")
    blob.upload_from_filename(str(local_path))
    print(f"📤 Uploaded to gs://{bucket.name}/{folder}/{Path(local_path).name}")

set_global_variable("text_embedding_model", text_embedding_model)
set_global_variable("multimodal_embedding_model", multimodal_embedding_model)