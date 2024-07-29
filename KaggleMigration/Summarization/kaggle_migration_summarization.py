import os
import zipfile
import json
import glob
import io

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
            

class KaggleMigrationSummarizer:
    """Generation of information about collection, structure, and usecases of a dataset"""

    def __init__(self) -> None:
        """
        Initializes a new instance of the class.

        This constructor initializes the `dataset_prompt` attribute to `None`.

        Parameters:
            None

        Returns:
            None
        """
        self.dataset_prompt = None
        self.llm = LLM()

    def _compose_metadata_prompt(
        self,
        json_file: io.TextIOWrapper
    ) -> str:
        """
        Creates a prompt containing dataset metadata.

        Metadata includes ID, Title, Subtitle, Description, Keywords.

        Parameters:
            json_file: Dataset metadata file from Kaggle in the format of JSON

        Returns:
            str
        """
        metadata_prompt = ""
        metadata_json = json.load(json_file)
        
        if 'id' in metadata_json and metadata_json['id'] != '':
            metadata_prompt += f"ID ДАТАСЕТА:\n{metadata_json['id'][:1024]}\n\n"
        
        if 'hasTitle' in metadata_json and metadata_json['hasTitle'] is True:
            if 'title' in metadata_json and metadata_json['title'] != '':
                metadata_prompt += f"ЗАГОЛОВОК ДАТАСЕТА:\n{metadata_json['title'][:1024]}\n\n"
        
        if 'hasSubtitle' in metadata_json and metadata_json['hasSubtitle'] is True:
            if 'subtitle' in metadata_json and metadata_json['subtitle'] != '':
                metadata_prompt += f"ПОДЗАГОЛОВОК ДАТАСЕТА:\n{metadata_json['subtitle'][:1024]}\n\n"
        
        if 'hasDescription' in metadata_json and metadata_json['hasDescription'] is True:
            if 'description' in metadata_json and metadata_json['description'] != '':
                metadata_prompt += f"ОПИСАНИЕ ДАТАСЕТА:\n{metadata_json['subtitle'][:1024]}\n\n"
        
        if 'keywords' in metadata_json and len(metadata_json['keywords']) != 0:
            metadata_prompt += f"КЛЮЧЕВЫЕ СЛОВА ДАТАСЕТА:\n{str(metadata_json['keywords'])[:1024]}\n\n"
        
        return metadata_prompt

    def _compose_data_prompt(
        self,
        zip_file: zipfile.ZipFile
    ) -> str:
        """
        Creates a prompt containing pieces of dataset files.

        Resulting data prompt includes only 4 files from the dataset at most (if there
        are more files, the total number of files is mentioned). Any extension
        of the file could be processed. Only the first 1024 bytes of each file is included. 

        Parameters:
            zip_file: ZIP archive of the dataset files from Kaggle

        Returns:
            str
        """
        data_prompt = "СОДЕРЖИМОЕ ФАЙЛОВ ДАТАСЕТА"
        infolist_files = [file_info for file_info in zip_file.infolist() if not file_info.is_dir()]
    
        if len(infolist_files) > 4:
            data_prompt += f" (показано только 4 файла из {len(file_paths)})"
    
        data_prompt += ":\n"
        
        # max 4 files
        for file_info in infolist_files[:4]:
            
            data_prompt += f"Имя файла: {file_info.filename}\n"
    
            with zip_file.open(file_info) as f:
                data_file_content = f.read(1024)
    
            data_prompt += f"Первые 1024 байт файла: {data_file_content}\n\n"
    
        return data_prompt
    
    def _compose_dataset_prompt(
        self,
        json_file: io.TextIOWrapper,
        zip_file: zipfile.ZipFile
    ) -> str:
        """
        Creates a prompt containing dataset metadata and pieces of files.

        The resulting dataset prompt is a concatenation of metadata prompt
        and data prompt

        Parameters:
            json_file: Dataset metadata file from Kaggle in the format of JSON
            zip_file: ZIP archive of the dataset files from Kaggle

        Returns:
            str
        """
        metadata_prompt = self._compose_metadata_prompt(json_file)
        data_prompt = self._compose_data_prompt(zip_file)
        return metadata_prompt + data_prompt

    def _compose_collection_method_prompt(
        self,
        json_file: io.TextIOWrapper,
        zip_file: zipfile.ZipFile
    ) -> str:
        """
        Creates a prompt for model to generate description about how
        the dataset was collected.

        Parameters:
            json_file: Dataset metadata file from Kaggle in the format of JSON
            zip_file: ZIP archive of the dataset files from Kaggle

        Returns:
            str
        """

        # create a dataset prompt if wasn't created before 
        if self.dataset_prompt is None:
            self.dataset_prompt = self._compose_dataset_prompt(
                json_file, zip_file
            )
        
        # load dataset prompt previously created
        dataset_prompt = self.dataset_prompt

        dataset_prompt += "Cоставь один абзац текста описывающий как были собраны данные. "
        dataset_prompt += "Если ты не знаешь как были собраны данные, то в ответе напиши, что нет информации о том, как были собраны данные. "
        dataset_prompt += "Не упоминай заголовок и ID датасета. "
        dataset_prompt += "Не упоминай файлы датасета. "
        dataset_prompt += "Не говори о ключевых словах датасета. "
        dataset_prompt += "Не упоминай себя. "
        dataset_prompt += "Не говори для чего датасет может быть полезен. "
        dataset_prompt += "Сформулируй ответ в одном абзаце на русском языке."
        
        return dataset_prompt

    def _compose_dataset_structure_prompt(
        self,
        json_file: io.TextIOWrapper,
        zip_file: zipfile.ZipFile
    ) -> str:
        """
        Creates a prompt for model to generate description about the structure
        of the dataset.

        Parameters:
            json_file: Dataset metadata file from Kaggle in the format of JSON
            zip_file: ZIP archive of the dataset files from Kaggle

        Returns:
            str
        """

        # create a dataset prompt if wasn't created before 
        if self.dataset_prompt is None:
            self.dataset_prompt = self._compose_dataset_prompt(
                json_file, zip_file
            )

        # load dataset prompt previously created
        dataset_prompt = self.dataset_prompt

        dataset_prompt += "Cоставь один абзац текста описывающий структуру датасета. "
        dataset_prompt += "Не упоминай ID датасета, заголовок и подзаголовок. "
        dataset_prompt += "Не упоминай фразу \"первые 1024 байт датасета\". "
        dataset_prompt += "Говори только собственное имя файла, игнорируя полный путь. "
        dataset_prompt += "Не рассказывай как был собран датасет. "
        dataset_prompt += "Не говори о ключевых словах датасета. "
        dataset_prompt += "Не упоминай себя. "
        dataset_prompt += "Не говори для чего датасет может быть полезен. "
        dataset_prompt += "Сформулируй ответ в одном абзаце на русском языке."

        return dataset_prompt

    def _compose_usecases_prompt(
        self,
        json_file,
        zip_file
    ) -> str:
        if self.dataset_prompt is None:
            self.dataset_prompt = self._compose_dataset_prompt(
                json_file, zip_file
            )

        dataset_prompt = self.dataset_prompt

        dataset_prompt += "Cоставь один абзац текста описывающий возможные варианты использования датасета. "
        dataset_prompt += "Не упоминай ID датасета, заголовок и подзаголовок. "
        dataset_prompt += "Не рассказывай как был собран датасет. "
        dataset_prompt += "Не говори о ключевых словах датасета. "
        dataset_prompt += "Не упоминай себя. "
        dataset_prompt += "Сформулируй ответ в одном абзаце на русском языке. "
        
        return dataset_prompt

    def _generate(
        self,
        prompt: str
    ) -> str:
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
    
    def generate_collection_description(self, json_file, zip_file):
        """
        Make a request LLM to generate a description about how dataset was collected.

        Parameters:
            json_file: Dataset metadata file from Kaggle in the format of JSON
            zip_file: ZIP archive of the dataset files from Kaggle

        Returns:
            str
        """

        prompt = self._compose_collection_method_prompt(json_file, zip_file)
        response = self._generate(prompt)
        return response

    def generate_structure_description(
        self,
        json_file: io.TextIOWrapper,
        zip_file: zipfile.ZipFile
    ) -> str:
        """
        Make a request LLM to generate a description about dataset structure.

        Parameters:
            json_file: Dataset metadata file from Kaggle in the format of JSON
            zip_file: ZIP archive of the dataset files from Kaggle

        Returns:
            str
        """

        prompt = self._compose_dataset_structure_prompt(json_file, zip_file)
        response = self._generate(prompt)
        return response

    def generate_usecases_description(
        self,
        json_file: io.TextIOWrapper,
        zip_file: zipfile.ZipFile
    ) -> str:
        """
        Make a request LLM to generate a description about potential uses of the dataset.

        Parameters:
            json_file: Dataset metadata file from Kaggle in the format of JSON
            zip_file: ZIP archive of the dataset files from Kaggle

        Returns:
            str
        """

        prompt = self._compose_usecases_prompt(json_file, zip_file)
        response = self._generate(prompt)
        return response

    def generate_descriptions(
        self,
        json_file: io.TextIOWrapper,
        zip_file: zipfile.ZipFile
    ) -> str:
        """
        Make a requests LLM to generate three descriptions: collection method, dataset
        structure, and usecases.

        Parameters:
            json_file: Dataset metadata file from Kaggle in the format of JSON
            zip_file: ZIP archive of the dataset files from Kaggle

        Returns:
            dict
        """

        collection_description = self.generate_collection_description(json_file, zip_file)
        structure_description = self.generate_structure_description(json_file, zip_file)
        usecases_description = self.generate_usecases_description(json_file, zip_file)
        return {
            'collection_description': collection_description,
            'structure_description': structure_description,
            'usecases_description': usecases_description
        }

# Usage Example
if __name__ == "__main__":
    summarizer = KaggleMigrationSummarizer()
    json_path = os.path.join('..', 'InputExample', 'dataset-metadata.json')
    zip_path = os.path.join('..', 'InputExample', 'hotel-recommendation-dataset.zip')

    with open(json_path) as json_file:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            print(summarizer.generate_descriptions(json_file, zip_file))
