import json

# input_filepath = "/home/aix23606/seoah/JejuDialectSTT/train_1인따라말하기/dataset_1인 따라말하기.json"
# outputh_filepath = "/home/aix23606/seoah/JejuDialectSTT/train_1인따라말하기/dataset_1인 따라말하기_update.json"
#
# original_path = "C:/Users/김우영/Desktop/2024-2/딥러닝/project/Trimmed_data"
# replacement_path = "/home/aix23606/seoah/JejuDialectSTT/train_1인따라말하기/Train_data_1인 따라말하기"

input_filepath = "/home/aix23606/seoah/JejuDialectSTT/train_1인따라말하기/dataset_1인 따라말하기.json"
outputh_filepath = "/home/aix23606/seoah/JejuDialectSTT/train_1인따라말하기/mod_dataset_1인 따라말하기.json"

original_path = "C:/Users/김우영/Desktop/2024-2/딥러닝/project/Trimmed_data"
replacement_path = "/home/aix23606/seoah/JejuDialectSTT/train_1인따라말하기/Train_data_1인 따라말하기"


with open(input_filepath, "r", encoding="utf-8") as file:
    data = json.load(file)

for item in data["dataset"]:
    item["audio_filepath"] = item["audio_filepath"].replace(original_path, replacement_path)

with open(outputh_filepath, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print("Done!")

