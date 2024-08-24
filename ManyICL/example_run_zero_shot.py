import pandas as pd
import numpy as np
import openai
import tenacity
from PIL.Image import Image
import base64
from abc import ABC, abstractmethod
import re
import asyncio
import time
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

openaikey = os.getenv('OPENAI_API_KEY')

class UnanswerableError(Exception):
    """An exception indicating the prompt was too long for the model."""

def parse_answers(text):
    pattern = r"<ANS>\s*([A-N, ]+)\s*</ANS>"
    match = re.search(pattern, text)
    letter_to_index = {chr(i): i - ord('A') for i in range(ord('A'), ord('N') + 1)}
    vector = [0] * 14
    
    if match:
        letters = match.group(1).split(',')
        for letter in letters:
            letter = letter.strip()
            if letter in letter_to_index:
                vector[letter_to_index[letter]] = 1
    
    return vector

class OpenAIModel(ABC):
    def __init__(self, model_kwargs: dict, is_async=False, **kwargs):
        super().__init__(**kwargs)
        self.model_kwargs = model_kwargs.copy()
        self.model_kwargs.setdefault("model", "gpt-4o")
        self.detail = "high"
        if is_async:
            self.client = openai.AsyncOpenAI(api_key=openaikey)
        else:
            self.client = openai.OpenAI(api_key=openaikey)
        
    def generate_text_url(self, text):
        return {"type": "text", "text": text}
        
    def generate_image_url(self, image_path, detail="low"):
        def encode_image(image_path):
            if str(image_path).lower().endswith("tif"):
                with Image.open(image_path) as img:
                    img.convert("RGB").save("temp.jpeg", "JPEG")
                image_path = "temp.jpeg"
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64, {encode_image(image_path)}",
                "detail": detail,
            },
        }

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),  # Increase the number of retries
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),  # Increase wait time between retries
        retry=tenacity.retry_if_exception(
            lambda exc: not isinstance(exc, UnanswerableError)
        ),
    )
    async def get_completion_async(self, text_prompt: str, test_cxr: str, demo_cxr: list[str] | None) -> str:
        if demo_cxr:
            raise NotImplementedError("ICL not implemented yet")
        else:
            messages = []
            messages.append(self.generate_image_url(test_cxr, detail=self.detail))
            messages.append(self.generate_text_url(text_prompt.split("<<IMG>>")[1]))
        try:
            response = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": messages}],
                **self.model_kwargs,
            )
        except openai.BadRequestError as e:
            if "PromptTooLongError" in e.message:
                raise UnanswerableError(e.message) from e
            raise

        return response.choices[0].message.content
    
    def get_completion(self, text_prompt: str, test_cxr: str, demo_cxr: list[str] | None) -> str:
        if demo_cxr:
            raise NotImplementedError("ICL not implemented yet")
        else:
            messages = []
            messages.append(self.generate_image_url(test_cxr, detail=self.detail))
            messages.append(self.generate_text_url(text_prompt.split("<<IMG>>")[1]))
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": messages}],
                **self.model_kwargs,
            )
        except openai.BadRequestError as e:
            if "PromptTooLongError" in e.message:
                raise UnanswerableError(e.message) from e
            raise

        return response.choices[0].message.content
    
async def process_row(model, row, semaphore):
    async with semaphore:
        ground_truth_vec = (row.iloc[5:19] == 1).values.astype(float)
        race = 'White' if row.race in ['White', 'White, non-Hispanic', 'White or Caucasian'] else 'Black'
        age = row.Age
        sex = row.Sex
        path = row.updated_path

        formatted_columns = ", ".join([f"{chr(65+i)}. {col}" for i, col in enumerate(row.index[5:19])])
        prompt = f"""<<IMG>>Given the image above, answer the following question using the specified format. 
        Question: Which of the following radiographic findings are present in the image above? More than one finding may be present per image.
        Choices: {formatted_columns}

        Please respond with the following format for each question, in the form of a comma delimited list of capital letters specifying which radiographic findings are present in the image surrounded by beginning <ANS> and end </ANS> brackets:
        ---BEGIN FORMAT TEMPLATE FOR QUESTION---
        <ANS> Your comma-delimited list of capital letters representing radiographic findings here </ANS>
        ---END FORMAT TEMPLATE FOR QUESTION---

        Do not deviate from the above format, because it will be parsed with the following regex r"<ANS>\s*([A-N, ]+)\s*</ANS>". Repeat the format template for the answer."""

        response = await model.get_completion_async(prompt, path, None)
        ans = parse_answers(response)

        return {
            "response": response,
            "parsed_answer": ans,
            "age": age,
            "sex": sex,
            "path": path,
            "ground_truth": ground_truth_vec,
            "race": race
        }

async def process_dataframe_async(model, test_frame, semaphore):
    tasks = []
    for _, row in test_frame.iterrows():
        task = process_row(model, row, semaphore)
        tasks.append(task)
        
    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing rows"):
        result = await task
        results.append(result)

    result_df = pd.DataFrame(results)
    return result_df

async def main():
    test_frame = pd.read_csv('/home/joseph/datasets/chexpertchestxrays-u20210408/chexpert_test_df_labels.csv', index_col=0)
    async_model = OpenAIModel({"model": "gpt-4o"}, is_async=True)
    
    # Define semaphore with a limit of 20 concurrent requests
    semaphore = asyncio.Semaphore(10)

    # Time async processing
    start_time_async = time.time()
    output_frame_async = await process_dataframe_async(async_model, test_frame, semaphore)
    output_frame_async.to_csv('full_output_frame_zero_shot_high_res.csv')
    end_time_async = time.time()
    async_duration = end_time_async - start_time_async

    print(f"Async processing time: {async_duration:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
