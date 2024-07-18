import ollama
import requests

class SummaryGenerator:
    def __init__(self):
        pass

    def generate_summary(self, collection_info, structure_info, usecases_info):
        """Generates a summary based on collected information."""
        prompt = self._compose_full_prompt(collection_info, structure_info, usecases_info)
        
        try:
            result = ollama.generate(model='saiga', prompt=prompt)
            response = result['response']
            return response.strip()
        except requests.exceptions.RequestException as e:
            print(f"Error in API request: {e}")
            return "Error generating summary"
        
        Regeneration

    def _compose_full_prompt(self, collection_info, structure_info, usecases_info):
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

        prompt += ("Теперь, пожалуйста, суммаризируй эти три абзаца в один небольшой абзац на русском языке, "
                   "уделяя равное внимание методу сбора данных, структуре данных и потенциальным юзкейсам.")

        return prompt

if __name__ == "__main__":
    summary_generator = SummaryGenerator()

    collection_info = input("Введите информацию о методе сбора данных: ")
    structure_info = input("Введите информацию о структуре данных: ")
    usecases_info = input("Введите информацию о потенциальных юзкейсах: ")

    summary = summary_generator.generate_summary(collection_info, structure_info, usecases_info)
    print("Суммаризация: ", summary)