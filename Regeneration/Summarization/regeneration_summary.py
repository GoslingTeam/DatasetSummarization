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

class SummaryGenerator:
    def __init__(self):
        self.llm = LLM()

    def generate_summary(self, collection_info, structure_info, usecases_info):
        """Generates a summary based on collected information."""
        prompt = self._compose_full_prompt(collection_info, structure_info, usecases_info)
        response = self._generate(prompt)
        return response

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
        return response['response'].strip()

if __name__ == "__main__":
    summary_generator = SummaryGenerator()

    collection_info = 'Датасет YTS movie dataset содержит информацию о фильмах и телесериалах, включая их названия, год выпуска, рейтинг IMDB, жанры, режиссеров, актёров иsimilar movies. В датасете представлено множество фильмов на английском языке, относящихся к разным жанрам: комедийные, криминальные, драматические, мюзиклы и другие. Например, в одном из файлов представлены фильмы "All In: The Family" (2020), "The Razors Edge" (1984) и "Bebes Kids" (1992).Данные были собраны неизвестным автором, и информация о сборе данных не предоставляется. Однако можно предположить, что данные были взяты из различных источников, таких как IMDb, Rotten Tomatoes и других сайтов для киноафиши, и были проанализированы с целью анализа и рекомендации фильмов. В целом, данный датасет может быть полезен для аналитиков и разработчиков систем рекомендаций в области кинематографии и телевидения.'
    structure_info = 'Датасет "YTS movie dataset" представляет собой сборник данных для анализа и системы рекомендаций фильмов. Данные содержатся в файле JSON и включают информацию о различных фильмах, таких как название, год выпуска, оценка IMDb, жанры, названия актеров и режиссеров, а также похожие фильмы. Файл разделен на два файла: yts.json и yts_2022.json. Они содержат информацию о различных фильмах, включая их характеристики и similarities.'
    usecases_info = 'Датасет YTS movie dataset - это крупное многофункциональное коллективное исследование и рекомендационная система, которое содержит данные о фильмах и их аналогичных атрибутах для изучения, оценки и подбора нового контента. Эта база данных может быть использована в различных областях, таких как аналитика и отчетность, машинное обучение и прогнозирование, поиск информации и рекомендации. Например, можно использовать этот датасет для анализа и изучения предпочтений зрителей, определения тенденций и трендов в киноиндустрии, исследования влияния различных жанров и режиссеров на оценку фильмов. Также данный датасет может быть использован для создания рекомендационных систем, которые помогут пользователям выбрать новый интересный фильм или сериал в соответствии с их вкусовыми предпочтениями.Кроме того, YTS movie dataset может быть применен для поиска информации о кинофильмах и анализа их качественных характеристик, таких как жанр, год выхода, оценка IMDb и т.д. Это особенно полезно в случае, если необходимо найти фильм с определенным набором признаков или жанров.В целом, данный датасет предоставляет широкие возможности для анализа и использования в различных сферах, от науки до бизнеса, и может быть использован как для изучения и анализа киноиндустрии, так и для создания инновационных решений и систем.'

    summary = summary_generator.generate_summary(collection_info, structure_info, usecases_info)
    print("Суммаризация: ", summary)