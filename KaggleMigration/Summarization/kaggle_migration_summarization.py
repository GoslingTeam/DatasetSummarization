import os
import zipfile
import json
import glob
import ollama

class KaggleMigrationSummarizer:
    """Generation of information about collection, structure, and usecases of a dataset"""

    def __init__(self):
        """
        Initializes a new instance of the class.

        This constructor initializes the `dataset_prompt` attribute to `None`.

        Parameters:
            None

        Returns:
            None
        """
        self.dataset_prompt = None

    def _compose_dataset_prompt(self, dataset_dir_name):
        """
        Composes a prompt for a dataset given its directory name.
        This function iterates over the files in the specified dataset directory and composes a prompt based on the metadata and content of the files.
        
        Parameters:
            dataset_dir_name (str): The name of the directory containing the dataset.

        Returns:
            str: The composed prompt for the dataset.
        """

        for filename in os.listdir(dataset_dir_name):

            if filename.endswith('.json'):

                metadata_prompt = ""

                with open(os.path.join(dataset_dir_name, filename), encoding='UTF-8') as f:
                    metadata_json = json.load(f)

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


            if filename.endswith('.zip'):

                data_prompt = "СОДЕРЖИМОЕ ФАЙЛОВ ДАТАСЕТА"

                extraction_dir_name = os.path.join(dataset_dir_name, filename[:-4])

                if not os.path.exists(extraction_dir_name):
                    os.makedirs(extraction_dir_name)
                    with zipfile.ZipFile(os.path.join(dataset_dir_name, filename), 'r') as zip_ref:
                        zip_ref.extractall(extraction_dir_name)

                all_paths = glob.glob(extraction_dir_name + "/**/*", recursive=True)
                file_paths = [path for path in all_paths if os.path.isfile(path)]

                if len(file_paths) > 4:
                    data_prompt += f" (показано только 4 файла из {len(file_paths)})"

                data_prompt += ":\n"

                # max number of files is 4
                for data_filename in file_paths[:4]:

                    data_prompt += f"Имя файла: {data_filename}\n"

                    with open(data_filename, 'rb') as f:
                        data_file_content = f.read(1024)

                    data_prompt += f"Первые 1024 байт файла: {data_file_content}\n\n"

        prompt = metadata_prompt + data_prompt + "\n"
        return prompt

    def _compose_collection_method_prompt(self, dataset_dir_name):
        """
        Composes a prompt for the collection method of a dataset.

        Args:
            dataset_dir_name (str): The name of the directory containing the dataset.

        Returns:
            str: The composed prompt for the collection method of the dataset.
        """
        if self.dataset_prompt is None:
            self.dataset_prompt = self._compose_dataset_prompt(dataset_dir_name)

        dataset_prompt = self.dataset_prompt

        dataset_prompt += "Используй информацию из секций, обозначенных ЗАГЛАВНЫМИ буквами, чтобы составить один абзац текста описывающий "
        dataset_prompt += "как данные из датасета были собраны. Если в секциях выше нет информации о том, как данные были собраны, то просто сообщи об этом. "
        dataset_prompt += "Свой ответ составь на русском языке."
        
        return dataset_prompt

    def _compose_dataset_structure_prompt(self, dataset_dir_name):
        """
        Composes a prompt for the structure of a dataset.

        Args:
            dataset_dir_name (str): The name of the directory containing the dataset.

        Returns:
            str: The composed prompt for the structure of the dataset.
        """
        if self.dataset_prompt is None:
            self.dataset_prompt = self._compose_dataset_prompt(dataset_dir_name)

        dataset_prompt = self.dataset_prompt

        dataset_prompt += "Используй информацию из секций, обозначенных ЗАГЛАВНЫМИ буквами, чтобы составить один абзац текста описывающий "
        dataset_prompt += "структуру датасета. Если в секциях выше недостаточно информации о структуре датасета, то просто сообщи об этом. "
        dataset_prompt += "Свой ответ составь на русском языке."

        return dataset_prompt

    def _compose_usecases_prompt(self, dataset_dir_name):
        """
        Composes a prompt for the use cases of a dataset.

        Parameters:
            dataset_dir_name (str): The name of the directory containing the dataset.

        Returns:
            str: The composed prompt for the use cases of the dataset.
        """
        if self.dataset_prompt is None:
            self.dataset_prompt = self._compose_dataset_prompt(dataset_dir_name)

        dataset_prompt = self.dataset_prompt

        dataset_prompt += "Используй информацию из секций, обозначенных ЗАГЛАВНЫМИ буквами, чтобы составить один абзац текста описывающий "
        dataset_prompt += "возможные варианты использования датасета. "
        dataset_prompt += "Свой ответ составь на русском языке."
        
        return dataset_prompt
    
    def generate_collection_description(self, dataset_dir_name):
        """
        Generates a collection description based on the dataset directory name.

        Args:
            dataset_dir_name (str): The directory name of the dataset.

        Returns:
            str: The generated collection description.
        """
        prompt = self._compose_collection_method_prompt(dataset_dir_name)
        result = ollama.generate(model='bambucha/saiga-llama3', prompt=prompt)
        response = result['response']
        response = response.strip()
        return response

    def generate_structure_description(self, dataset_dir_name):
        """
        Generates a structure description for a dataset based on the given dataset directory name.

        Parameters:
            dataset_dir_name (str): The name of the dataset directory.

        Returns:
            str: The generated structure description.
        """
        prompt = self._compose_dataset_structure_prompt(dataset_dir_name)
        result = ollama.generate(model='bambucha/saiga-llama3', prompt=prompt)
        response = result['response']
        response = response.strip()
        return response

    def generate_usecases_description(self, dataset_dir_name):
        """
        Generates a structure description for a dataset based on its directory name.

        Args:
            dataset_dir_name (str): The directory name of the dataset.

        Returns:
            str: The generated structure description for the dataset.
        """
        prompt = self._compose_usecases_prompt(dataset_dir_name)
        result = ollama.generate(model='bambucha/saiga-llama3', prompt=prompt)
        response = result['response']
        response = response.strip()
        return response
    
if __name__ == "__main__":
    summarizer = KaggleMigrationSummarizer()
    folder = '../aarzookuhar/hotel-recommendation-dataset'
    print('COLLECTION')
    print(summarizer.generate_collection_description(folder))
    print()
    print('STRUCTURE')
    print(summarizer.generate_structure_description(folder))
    print()
    print('USECASES')
    print(summarizer.generate_usecases_description(folder))

