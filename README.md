# RPG-game
ChatGPT said:  This Python RPG blends classic gameplay with AI-driven progression. Using TinyLlama fine-tuning or GPT-4o, the game dynamically upgrades levels, balances challenges, and generates creative events, making each adventure unique, adaptive, and engaging while enhancing storytelling and player experience.

This project is built on python 3.11.0
Create a new project with costum python 3.11.0 virtual environment
Check if all the relevent libraries are installed
Before running the program, go to Test_Files and run tinyllama to download TinyLlama-1.1B-intermediate-step-1431k-3T as tinyllama
Use cmd : setx OPENAI_API_KEY "api-key" to initiate the API key
Check if CUDA is enabled and pytorch is installed with 'cuda121' or higher (This program has been tested on cuda126)
Run api_test and cuda_test to verify if they work properly
Set the file paths very carefully in "sample_generator", "Level_generator", "FineTuner" and "dataset_preprocessor" files. Do not change their last directory name + format.
Note: sample_instance.json and trmporary_dataset.json under self.training_dataset, self.temporary_dataset.json files under Preprocessor class -> dataset_preprocessor.py are deleted and recreated during the program execution. Therefore they might not be visible at times.
GPT_progress file contain the number of failed requests vs successes of gpt-4o which currently runs at approx. 13%
Note: Level generation time may depend on the PC and networking conditions.
Run RPGIntellignt.py to execute the program.
If a large dataset needs to be collected, run dataset_collector.py by importing it to the main directory.
