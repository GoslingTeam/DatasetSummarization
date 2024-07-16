import os
import json
import ollama
import requests

class DatasetProcessor:
    def init(self, base_dir):
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, 'RegenerationDataset', 'Responses')
        self.dataset_dirs = {
            'collection': os.path.join(base_dir, 'CollectionMethodResponses'),
            'structure': os.path.join(base_dir, 'DatasetStructureResponses'),
            'usecases': os.path.join(base_dir, 'PotentialUsesResponses')
        }

    def load_info_from_folder(self, folder_path):
        """Loads information from a specific folder."""
        info = {}

        if os.path.isfile(folder_path):
            with open(folder_path, encoding='UTF-8') as f:
                content = f.read()
            info[os.path.basename(folder_path)] = content
        elif os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if filename.endswith('.txt'):
                    with open(file_path, encoding='UTF-8') as f:
                        content = f.read()
                    info[filename] = content

        return info

    def compose_full_prompt(self, collection_info, structure_info, usecases_info):
        """Composes a full prompt including existing responses."""
        prompt = ""

        if collection_info:
            prompt += f"Метод сбора данных:\n{collection_info}\n\n"
        else:
            prompt += "Метод сбора данных:\nИнформация о методе сбора данных отсутствует.\n\n"

        if structure_info:
            prompt += f"Структура данных:\n{structure_info}\n\n"
        else:
            prompt += "Структура данных:\nИнформация о структуре данных отсутствует.\n\n"

        if usecases_info:
            prompt += f"Потенциальные юзкейсы:\n{usecases_info}\n\n"
        else:
            prompt += "Потенциальные юзкейсы:\nИнформация о потенциальных юзкейсах отсутствует.\n\n"

        prompt += "Теперь суммаризируй эти три абзаца в один небольшой абзац на русском языке."

        return prompt

    def generate_summary(self, collection_info, structure_info, usecases_info):
        """Generates a summary based on collected information."""
        prompt = self.compose_full_prompt(collection_info, structure_info, usecases_info)
        
        try:
            result = ollama.generate(model='saiga', prompt=prompt)
            response = result['response']
            return response.strip()
        except requests.exceptions.RequestException as e:
            print(f"Error in API request: {e}")
            return "Error generating summary"

    def process_all_datasets(self):
        """Processes all datasets in the given base directory."""
        os.makedirs(self.output_dir, exist_ok=True)

        for dataset_type, folder_path in self.dataset_dirs.items():
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    dataset_name = os.path.splitext(filename)[0]
                    output_dataset_dir = os.path.join(self.output_dir, dataset_name)
                    os.makedirs(output_dataset_dir, exist_ok=True)

                    collection_info = ''
                    structure_info = ''
                    usecases_info = ''

                    if dataset_type == 'collection':
                        collection_info = self.load_info_from_folder(os.path.join(folder_path, filename)).get(filename, '')
                    elif dataset_type == 'structure':
                        structure_info = self.load_info_from_folder(os.path.join(folder_path, filename)).get(filename, '')
                    elif dataset_type == 'usecases':
                        usecases_info = self.load_info_from_folder(os.path.join(folder_path, filename)).get(filename, '')

                    summary = self.generate_summary(collection_info, structure_info, usecases_info)
                    output_file_path = os.path.join(output_dataset_dir, filename)
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        f.write(summary)
                    print(f"Summary written to: {output_file_path}")

def main(base_dir):
    processor = DatasetProcessor(base_dir)
    processor.process_all_datasets()
    print(f"Summaries have been written to {processor.output_dir}")

if __name__ == "main":
    base_dir = '/Users/zihadeev/Downloads/DatasetSummarization-main/KaggleMigrationDataset'
    main(base_dir)