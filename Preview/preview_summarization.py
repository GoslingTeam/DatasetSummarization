import os
import zipfile
import json
import glob

import aiohttp
import asyncio

class LLM:
    """Class for making requests to LLM endpoint"""

    def __init__(self):
        """
        Constructor that reads the URL to the endpoint from a file. The file should
        be in the same folder with this file.

        Parameters:
            None

        Returns:
            None
        """
        with open('llama3_70b_endpoint.txt') as f:
            self.url: str = f.read()

    async def get_response(self, json_data) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=json_data) as response:
                if response.status != 200:
                    raise Exception(f"Error: {response.status}")
                return await response.json()

class PreviewSummarizer:

    def __init__(self) -> None:
        """
        Initializes a new instance of the class.

        Initializes a class for making requests to LLM.

        Parameters:
            None

        Returns:
            None
        """
        self.llm = LLM()

    def _compose_prompt(self, summary: str) -> str:
        """
        Compoases a prompt to LLM to compose a sentence.

        Parameters:
            summary: Summary of a dataset
        
        Returns:
            None
        """

        instruction = "Представь, что ты специалист по рекламе и любишь разговаривать лишь несколькими словами. "
        instruction += "Выше описан датасет, собранный другими людьми. "
        instruction += "Расскажи несколькими словами в чем особенность этого датасета, игнорируя упоминание названия датасета. "
        instruction += "Учти, что если ты напишешь, что это наш датасет, то ты будешь неправ. "
        instruction += "Сформулируй предложение так, чтобы им можно было подписать датасет на витрине датасетов. "
        instruction += "Напиши ответ в виде одного предложения на русском языке."

        prompt = f"{summary}/n/n{instruction}"
        return prompt

    def _generate(self, prompt: str) -> str:
        """
        Make a request to the deployed endpoint with LLaMA 3 70b to generate text. 

        Parameters:
            prompt: Prompt to the model

        Returns:
            str
        """

        response = asyncio.run(
            self.llm.get_response({
                "prompt": prompt,
                "stop": None,
                "max_tokens": 512,
                "choice": None,
                "schema": None,
                "regex": None,
                "temperature": 0.1,
            })
        )
        return response

    def generate_sentence(self, summary: str) -> str:
        """
        Composes a prompt and asks LLM to generate a sentence

        Parameters:
            summary: Summary of a dataset
        
        Returns:
            str
        """
        prompt = self._compose_prompt(summary)
        sentence = self._generate(prompt)
        return sentence

# Usage Example
if __name__ == "__main__":
    summary = 'Датасет "Coding Questions Solved With Code Llama 70B" содержит около 20 тысяч ответов искусственного интеллекта на задачи по программированию, подготовленные для обучения и тестирования алгоритмов кода. Метод сбора данных предполагает автоматическое генерирование ответов, что делает его уникальным источником информации о решениях на различные языки программирования, включая Python. Структура данных не предоставлена, однако в данных содержится множество примеров кода, которые могут вызывать различные результаты при выполнении. Эти данные потенциально могут быть полезны для обучения и тестирования алгоритмов кода, а также для лучшего понимания применения языков программирования в реальных ситуациях.'
    
    summarizer = PreviewSummarizer()
    print(summarizer.generate_sentence(summary))